"""RGB→丰度→HSI 解混训练与推理（Python 3 / PyTorch）。

运行：python Unmixing.py train --hsrs_dir ./dataset/tests
默认读取 dataset/trains 中的全部 Chikusei 样本用于训练；
已对齐为59波段的 HSRS 按固定随机种子划分80%验证、20%最终测试。
每次运行的 training.log 追加全部轮次和测试结果，验证 L1 最低的模型用于测试。
可传 --train_dirs /path/to/chikusei 和 --hsrs_dir /path/to/hsrs 指定数据目录。
外部 RGB 推理：python Unmixing.py infer --checkpoint /path/to/epoch_40.pth
    --input_dir ./dataset/aid_check9 --output_dir ./experiments/aid_check9/abundance --n_blocks 3
输出保留输入 MAT 名称，Abu 为五通道丰度，GT 为预处理后 RGB（非 HSI 真值）。
"""
import argparse
import os
import time
import numpy as np
import json
import scipy.io as sio

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

from unmixingmodel.unmixingAE import UnmixingAE
from core import utils
from core.common import *
from core.loaddata import HSIDataset, RGBDataset
from core.loss import reconstruction_SADloss,CharbonnierLoss,TVLossEndmembers
from core.metrics import quality_assessment, compare_sam, compare_mpsnr

# global settings
resume = False
log_interval = 50


def output_channels(dataset_name):
    return 59 if dataset_name == "Chikusei" else 31


def load_model_state(model, state_dict):
    """Load checkpoints saved with or without DataParallel prefixes."""
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(state_dict)


def forward_with_cudnn_fallback(model, inputs):
    """Run a forward pass and bypass cuDNN only for a sublibrary mismatch.

    cuDNN 9 is split into multiple DLLs.  A minimal convolution may succeed
    while a different convolution shape loads another, incompatible sublibrary.
    The fallback keeps CUDA tensors and uses PyTorch's native CUDA convolution;
    unrelated runtime errors are deliberately not intercepted.
    """

    try:
        return model(inputs)
    except RuntimeError as exc:
        mismatch = "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"
        cudnn_enabled = bool(getattr(torch.backends.cudnn, "enabled", False))
        if not inputs.is_cuda or not cudnn_enabled or mismatch not in str(exc):
            raise

        torch.backends.cudnn.enabled = False
        print(
            "WARNING: cuDNN sublibrary versions are inconsistent. "
            "cuDNN has been disabled and this forward pass will be retried "
            "with PyTorch's native CUDA convolution. Training may be slower."
        )
        return model(inputs)

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for AE network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=16, help="batch size, default set to 64")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="UnmixingAE", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED, help="fixed random seed")
    train_parser.add_argument("--learning_rate", type=float, default=2e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./experiments/unmixing/ckpts/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    train_parser.add_argument(
        "--skip_test",
        action="store_true",
        help="skip final test only; HSRS validation remains required",
    )

    train_parser.add_argument("--train_dirs", nargs="+", default=["./dataset/trains"],
                              help="Chikusei-only MAT directories (default: ./dataset/trains)")
    train_parser.add_argument("--hsrs_dir", default="./dataset/tests",
                              help="HSRS-SC MAT directory, spectrally aligned to 59 bands")

    infer_parser = subparsers.add_parser("infer", help="parser for inferring arguments")
    infer_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    infer_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    infer_parser.add_argument("--n_blocks", type=int, default=3, help="must match checkpoint; default 3")
    infer_parser.add_argument("--checkpoint", type=str, help="explicit checkpoint file; overrides ckpt_dir")
    infer_parser.add_argument("--input_dir", default="./dataset/train", help="flat RGB MAT directory")
    infer_parser.add_argument("--output_dir", default="./dataset/inferred_abu", help="abundance MAT output directory")
    infer_parser.add_argument("--ckpt_dir", type=str, default="./experiments/unmixing/ckpts/", help="dataset_name, default set to dataset_name")
    infer_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    infer_parser.add_argument("--model_title", type=str, default="UnmixingAE", help="model_title, default set to model_title")
    infer_parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED, help="fixed random seed")

    args = main_parser.parse_args()
    if args.subcommand is None:
        main_parser.error("specify either train or infer")
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        args.cuda = 0
    if args.subcommand == "train":
        train(args)
    else:
        infer(args)
    pass

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Start seed: ", args.seed)
    utils.set_random_seed(args.seed)

    print('===> Loading datasets')
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    colors = output_channels(args.dataset_name)
    # 默认只读取 trains；显式指定多个训练目录时合并读取，不移动原始数据。
    train_set = HSIDataset(args.train_dirs[0], augment=False, output_channels=colors)
    for directory in args.train_dirs[1:]:
        train_set.image_files.extend(
            HSIDataset(directory, augment=False, output_channels=colors).image_files
        )
    train_files = [os.path.realpath(path) for path in train_set.image_files]
    if len(set(train_files)) != len(train_files):
        raise ValueError("Duplicate Chikusei files in train_dirs")
    hsrs_set = HSIDataset(args.hsrs_dir, augment=False, output_channels=colors)
    if set(train_files) & {os.path.realpath(path) for path in hsrs_set.image_files}:
        raise ValueError("Training and HSRS directories must not overlap")
    if len(hsrs_set) < 2:
        raise ValueError("HSRS requires at least two samples for validation/test")
    # 独立 RNG 保证文件排序与 seed 相同时划分一致，不消费模型初始化 RNG。
    order = np.random.default_rng(args.seed).permutation(len(hsrs_set)).tolist()
    n_val = max(1, min(len(order) - 1, int(0.8 * len(order))))
    val_indices, test_indices = order[:n_val], order[n_val:]
    eval_set = Subset(hsrs_set, val_indices)
    test_set = Subset(hsrs_set, test_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=1, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('===> Building model')
    net = UnmixingAE(
        n_blocks=args.n_blocks,
        res_scale=0.1,
        input_channels=3,
        output_channels=colors,
        conv=default_conv
    )
    model_name = args.model_title + "_" + args.dataset_name +'_latest.pth'
    model_path = os.path.join(args.save_dir, model_name)
    
    if args.cuda and torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=device)
            start_epoch = checkpoint["epoch"]
            load_model_state(net, checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
    net.to(device).train()

    # loss functions to choose
    charbloss = CharbonnierLoss()
    SADLoss = reconstruction_SADloss()
    TVLoss = TVLossEndmembers()
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    log_dir  = 'experiments/unmixing/'+args.dataset_name + "_"+args.model_title+'_'+str(utils.get_timestamp())
    # 权重按运行隔离，避免下一次训练覆盖日志所指向的历史轮次。
    args.save_dir = os.path.join(args.save_dir, os.path.basename(log_dir))
    writer = SummaryWriter(log_dir)
    log_path = os.path.join(log_dir, "training.log")
    write_log(log_path, {
        "event": "configuration", "args": vars(args),
        "selection_metric": "minimum validation L1",
        "split": "HSRS sample-level fixed-seed 80/20; scene independence not verified",
        "train_files": train_files,
        "validation_files": [hsrs_set.image_files[i] for i in val_indices],
        "test_files": [hsrs_set.image_files[i] for i in test_indices],
    })
    best_loss, best_epoch = float("inf"), None
    best_name = args.model_title + "_" + args.dataset_name + "_best.pth"
    
    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e+1)
        epoch_start = time.perf_counter()
        totals = dict(total=0.0, charbonnier=0.0, weighted_sad=0.0, weighted_tv=0.0)
        sample_count = 0
        net.train()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (gt, rgbdata) in enumerate(train_loader):
            gt = gt.to(device)
            rgbdata = rgbdata.to(device)
            optimizer.zero_grad()       
            _, y, decoder_weight = forward_with_cudnn_fallback(net, rgbdata)

            charb_loss = charbloss(y,gt)
            sad_loss = 0.1 * SADLoss(y,gt)
            tv_endmembers = 0.015 * TVLoss(decoder_weight)
            loss = charb_loss +  sad_loss + tv_endmembers
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {e + 1}")
            batch_count = gt.shape[0]
            sample_count += batch_count
            for key, value in zip(totals, (loss, charb_loss, sad_loss, tv_endmembers)):
                totals[key] += value.item() * batch_count
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} \tEpoch[{}]({}/{}): Loss: {:.6f} charb_loss: {:.6f} SADLoss: {:.6f} TVLoss: {:.6f}".format(time.ctime(), args.n_blocks, e+1, iteration + 1,
                                                                   len(train_loader), loss.item(),charb_loss.item(), sad_loss.item(), tv_endmembers.item() ))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss.item(), n_iter)

        train_metrics = {key: value / sample_count for key, value in totals.items()}
        validation = validate(args, eval_loader, net, L1_loss)
        eval_loss = validation["L1"]
        if not np.isfinite(eval_loss):
            raise RuntimeError(f"Non-finite validation L1 at epoch {e + 1}")
        writer.add_scalar('scalar/avg_epoch_loss', train_metrics["total"], e + 1)
        for key, value in validation.items():
            writer.add_scalar('validation/' + key, value, e + 1)
        save_checkpoint(args, net, e + 1, model_name)
        model_t = args.model_title + "_" + args.dataset_name + "_epoch_" + str(e + 1) + ".pth"
        save_checkpoint(args, net, e + 1, model_t)
        if eval_loss < best_loss:
            best_loss, best_epoch = eval_loss, e + 1
            save_checkpoint(args, net, e + 1, best_name)
        # 一轮完成即追加并关闭文件，已完成轮次不会因后续中断丢失。
        write_log(log_path, {
            "event": "epoch", "epoch": e + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics, "validation": validation,
            "seconds": time.perf_counter() - epoch_start,
            "checkpoint": os.path.join(args.save_dir, model_t),
            "best_epoch": best_epoch, "best_validation_L1": best_loss,
        })

    write_log(log_path, {"event": "best_model", "epoch": best_epoch,
                         "validation_L1": best_loss,
                         "checkpoint": os.path.join(args.save_dir, best_name)})
    if args.skip_test:
        write_log(log_path, {"event": "test_skipped"})
    else:
        # Save the testing results only when an independent test set is available.
        print('===> Start testing best validation checkpoint')
        best_checkpoint = torch.load(os.path.join(args.save_dir, best_name), map_location=device)
        load_model_state(net, best_checkpoint["model"])
        net.to(device).eval()
        with torch.no_grad():
            output = []
            test_number = 0
            for i, (gt, rgbdata) in enumerate(test_loader):
                gt = gt.to(device)
                rgbdata = rgbdata.to(device)
                _, y, decoder_weight = forward_with_cudnn_fallback(net, rgbdata)
                y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
                y = y[:gt.shape[0],:gt.shape[1],:]
                if i == 0:
                    indices = quality_assessment(gt, y, data_range=1., ratio=1)
                else:
                    indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=1))
                output.append(y)
                test_number += 1
            for index in indices:
                indices[index] = indices[index] / test_number

        save_dir = os.path.join(log_dir, args.model_title + "_" + args.dataset_name + '_test.npy')
        np.save(save_dir, output)
        print("Test finished, test results saved to .npy file at ", save_dir)
        print(indices)

        write_log(log_path, {"event": "final_test", "epoch": best_epoch,
                             "samples": test_number, "metrics": indices})

    writer.close()


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def write_log(path, record):
    """将配置、每轮结果和最终测试追加到同一个可直接阅读的 UTF-8 日志。"""
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    if record["event"] != "configuration":
        print(line)


def validate(args, loader, model, criterion):
    """逐样本平均 L1、SAM（度）、MPSNR（dB）；仅 L1 参与最佳轮次选择。

    使用与训练相同的 HSI 归一化和伪 RGB，不裁剪重建值；无梯度更新。
    """
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval()
    totals = dict(L1=0.0, SAM=0.0, MPSNR=0.0)
    count = 0
    with torch.no_grad():
        for gt, rgbdata in loader:
            gt, rgbdata = gt.to(device), rgbdata.to(device)
            _, prediction, _ = forward_with_cudnn_fallback(model, rgbdata)
            for target, output in zip(gt, prediction):
                totals["L1"] += criterion(output, target).item()
                target = target.cpu().numpy().transpose(1, 2, 0)
                output = output.cpu().numpy().transpose(1, 2, 0)
                totals["SAM"] += compare_sam(target, output)
                totals["MPSNR"] += float(compare_mpsnr(target, output, data_range=1.0))
                count += 1
    model.train()
    return {key: value / count for key, value in totals.items()}


def infer(args):
    """使用指定解混权重推断 RGB MAT，保留样本名称并保存未裁剪的模型输出。"""
    utils.set_random_seed(args.seed)
    inferdata_path = args.input_dir
    result_path = args.output_dir
    if os.path.realpath(inferdata_path) == os.path.realpath(result_path):
        raise ValueError("Inference input and output directories must differ")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    colors = output_channels(args.dataset_name)
    inferdata_set = RGBDataset(image_dir=inferdata_path, augment=False)
    inferdata_loader = DataLoader(inferdata_set, batch_size=1, num_workers=4, shuffle=False)

    model_name = args.checkpoint or os.path.join(args.ckpt_dir, args.model_title + "_" + args.dataset_name + '_latest.pth')
    print(model_name)
    device = torch.device("cuda" if args.cuda else "cpu")
    ckpt = torch.load(model_name, map_location=device)["model"]
    net = UnmixingAE(
        n_blocks=args.n_blocks,
        res_scale=0.1,
        input_channels=3,
        output_channels=colors,
        conv=default_conv
    )
    load_model_state(net, ckpt)
    net.to(device).eval()

    print('===> Start inferring')
    with torch.no_grad():
        # loading model
        for i, rgbdata in enumerate(inferdata_loader):
            rgbdata = rgbdata.to(device)
            en_result, y, _ = forward_with_cudnn_fallback(net, rgbdata)
            # 保留原始数值，避免 clipping 掩盖重建越界或丰度异常。
            en_result = en_result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            rgbdata = rgbdata.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            y = y.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            source_mat = inferdata_set.image_files[i]
            filename = os.path.basename(source_mat)
            save_dir = os.path.join(result_path, filename)
            sio.savemat(save_dir, {'Abu': en_result, 'GT': rgbdata, 'Y': y,
                                  'source_mat': source_mat, 'checkpoint': model_name})

            if i % 100 == 0:
                print(i)

def save_checkpoint(args, model, epoch, ckpt_model_filename):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = args.save_dir
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {"epoch": epoch, "model": model_to_save.state_dict()}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))

if __name__ == "__main__":
    main()
