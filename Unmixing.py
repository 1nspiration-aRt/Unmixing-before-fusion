import argparse
import os
import time
import numpy as np
import json
import scipy.io as sio

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet import meter
from tensorboardX import SummaryWriter

from unmixingmodel.unmixingAE import UnmixingAE
from core import utils
from core.common import *
from core.loaddata import HSIDataset, RGBDataset
from core.loss import reconstruction_SADloss,CharbonnierLoss,TVLossEndmembers
from core.metrics import quality_assessment

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
    train_parser.add_argument("--n_feats", type=int, default=128, help="n_feats, default set to 256")
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

    infer_parser = subparsers.add_parser("infer", help="parser for inferring arguments")
    infer_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    infer_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    infer_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
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
    train_path    = './dataset/trains/'
    eval_path     = './dataset/evals/'
    test_data_dir = './dataset/tests/'

    colors = output_channels(args.dataset_name)
    train_set = HSIDataset(
        image_dir=train_path,
        augment=False,
        output_channels=colors
    )
    eval_set = HSIDataset(
        image_dir=eval_path,
        augment=False,
        output_channels=colors
    )
    test_set = HSIDataset(
        image_dir=test_data_dir,
        augment=False,
        output_channels=colors
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
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
    epoch_meter = meter.AverageValueMeter()
    log_dir  = 'experiments/unmixing/'+args.dataset_name + "_"+args.model_title+'_'+str(utils.get_timestamp())
    writer = SummaryWriter(log_dir)
    
    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e+1)
        epoch_meter.reset()
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
            epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} \tEpoch[{}]({}/{}): Loss: {:.6f} charb_loss: {:.6f} SADLoss: {:.6f} TVLoss: {:.6f}".format(time.ctime(), args.n_blocks, e+1, iteration + 1,
                                                                   len(train_loader), loss.item(),charb_loss.item(), sad_loss.item(), tv_endmembers.item() ))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss.item(), n_iter)

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e+1, epoch_meter.value()[0]))
        # run validation set every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        
        save_checkpoint(args, net, e+1, model_name)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 1 == 0:
            model_t = args.model_title + "_" + args.dataset_name +"_epoch_" + str(e+1) + ".pth"
            save_checkpoint(args, net, e+1, model_t)

    ## Save the testing results
    print('===> Start testing')
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
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=1)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=1))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    save_dir = os.path.join(log_dir , args.model_title + "_" + args.dataset_name + '_test.npy')
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr =os.path.join(log_dir , args.model_title + "_" + args.dataset_name +"_log.txt")
    json.dump(indices, open(QIstr, 'w'))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for _, (gt, rgbdata) in enumerate(loader):
            gt = gt.to(device)
            rgbdata = rgbdata.to(device)
            _, y, _ = forward_with_cudnn_fallback(model, rgbdata)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]

def infer(args):
    utils.set_random_seed(args.seed)
    inferdata_path  = './dataset/train/'
    result_path   = './dataset/inferred_abu/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    colors = output_channels(args.dataset_name)
    inferdata_set = RGBDataset(image_dir=inferdata_path, augment=False)
    inferdata_loader = DataLoader(inferdata_set, batch_size=1, num_workers=4, shuffle=False)

    model_name = os.path.join(args.ckpt_dir, args.model_title + "_" + args.dataset_name  +'_latest.pth')
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
            en_result = en_result.clamp_(*(0,1)).squeeze().cpu().numpy().transpose(1, 2, 0)
            rgbdata = rgbdata.clamp_(*(0,1)).squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y.clamp_(*(0,1)).squeeze().cpu().numpy().transpose(1, 2, 0)
            filename = str(i).zfill(4)
            save_dir = os.path.join(result_path, filename + '.mat')
            sio.savemat(save_dir,{'Abu':en_result, 'GT':rgbdata, 'Y':y})

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
