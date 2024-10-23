# Introduction

This is the implementation of article: Unmixing Before Fusion: A Generalized Paradigm for Multi-Source-based Hyperspectral Image Synthesis

# Usage

## Data

All the selected datasets are open-source. The experimental includes the remote sensing scenario and natural scenario. 

For the remote sensing scenario, the selected hyperspectral dataset includes the Chikusei and HSRS-SC dataset. The RGB dataset includes the AID and the NWPU dataset. 

For the natural sceneario, the selected hyperspectral dataset includes the ARAD and the Harvard dataset. The RGB dataset is the Place dataset. 

Before training the network, please download the dataset, crop the images and save as the .Mat file. It is recommended that the spatial size should not be larger than 256 x 256. The experimental workflow is as follows. 

# 1: Train the hyperspectral unmixing network and infer the abundance maps from external RGB dataset.

Input: Three channels images obtained by the band selection from original hyperspectral images.

Output: The original hyperspectral images.

Network: The trained RGB-HSI unmixing network.

For training the unmixing net, change the file path and run the following code.

python Unmixing.py train

After training, run the following code to get the abundance maps from external RGB datasets.

python Unmixing.py infer

After that, we can obtain the inferred abundance from multi-source datasets which include rich and realistic spatial distribution feature.

# 2: Train the Diffusion model and synthesize abundance maps by the trained Diffusion model.

Input: Inferred abundance maps by Step 1.

Output: Synthetic abundance maps.

For training the Diffusion model, please run:

python Diffusion.py -p train -c config/Chikusei_256_DDPM.json

For synthesizing abundance maps, please modify the 'resume_state' in the json file, and run:

python Diffusion.py -p val -c config/Chikusei_256_DDPM.json

After that, we can obtain the synthesized abundance in ./experiments/ddpm/\*/mat_results/.

# 3: HSI synthesis.

Input: Synthetic abundance maps by Step 2.

Output: Synthetic HSIs.

Change the train_path (path of synthesized abundances) and the model_name(the trained model of the unmixing net)

Run the following code to obtain the synthetic HSIs:

python Synthesis.py

After that, we can obtain the synthesized HSIs in ./experiments/Synthesis/HSI/ and corresponding RGB images in ./experiments/Synthesis/RGB/.


# Citation

Yu Y, Pan E, Wang X, et al. Unmixing Before Fusion: A Generalized Paradigm for Multi-Source-based Hyperspectral Image Synthesis[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 9297-9306.MLA
