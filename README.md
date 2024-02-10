# Mamba-UNet

Mamba-UNet: Unet-like Pure Visual Mamba for Medical Image Segmentation

The position of Mamba-UNet

<img src="netintro.png" width="50%" height="auto">


The position of Semi-Mamba-UNet

<img src="sslintro.png" width="50%" height="auto">


Mamba-UNet

<img src="framework.png" width="50%" height="auto">


## Mamba-UNet Zoo

Mamba-UNet -> [[Paper Link]](https://github.com/ziyangwang007/Mamba-UNet/blob/main/MambaUNet.pdf) Released in 6/Feb/2024.

Semi-Mamba-UNet -> [[Paper Link]](https://github.com/ziyangwang007/Mamba-UNet/blob/main/Semi_Mamba_UNet.pdf) Released in 10/Feb/2024.

3D Mamba-UNet -> TBA


## Requirements
* Pytorch, MONAI 
* Some basic python packages: Torchio, Numpy, Scikit-image, SimpleITK, Scipy, Medpy, nibabel, tqdm ......

```
cd casual-conv1d

python setup.py install
```

```
cd mamba

python setup.py install
```


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/Mamba-UNet.git 
cd Mamba-UNet
```

2. Download Pretrained Model

Download through [Google Drive](https://drive.google.com/file/d/14RzbbBDjbKbgr0ordKlWbb69EFkHuplr/view?usp=sharing) for SwinUNet, and [Google Drive](https://drive.google.com/file/d/1uUPsr7XeqayCxlspqBHbg5zIWx0JYtSX/view?usp=sharing) for Mamba-UNet. in `code/pretrained_ckpt'.

3. Download Dataset

Download through [Google Drive](https://drive.google.com/file/d/1F3JzBSIURtFJkfcExBcT6Hu7Ar5_f8uv/view?usp=sharing), and save in `data/ACDC'.

4. Train 2D UNet
```
python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/unet --model unet --max_iterations 10000
```

5. Train SwinUNet
```
python train_fully_supervised_2D_ViT.py --root_path ../data/ACDC --exp ACDC/swinunet --model swinunet --max_iterations 10000
```

6. Train Mamba-UNet
```
python train_fully_supervised_2D_VIM.py --root_path ../data/ACDC --exp ACDC/VIM --model mambaunet --max_iterations 10000
```

7. Train Semi-Mamba-UNet when 5% as labeled data
```
python train_Semi_Mamba_UNet.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet --max_iterations 30000 --labeled_num 3
```

8. Train Semi-Mamba-UNet when 10% as labeled data
```
python train_Semi_Mamba_UNet.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet --max_iterations 30000 --labeled_num 7
```

9. Test
```
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/xxx --model xxx
```

## Q&A

1. Q: Performance: I find my results are slightly lower than your reported results.

A: Please do not worry. The performance depends on many factors, such as how the data is split, how the network is initialized, and even the type of GPU used. What I want to emphasize is that you should maintain your hyper-parameter settings and test some other baseline methods. If method A has a lower Dice Coefficient than the reported number, it's likely that methods B and C will also have lower Dice Coefficients than the numbers reported in the paper.

2. Q: Concurrent Work: I found similar work about the integration of Mamba into UNet.

A: I am glad to see and acknowledge that there should be similar work. Mamba is a novel architecture, and it is obviously valuable to explore integrating Mamba with segmentation, detection, registration, etc. I am pleased that we all find Mamba efficient in some cases. This GitHub repository was developed on the 6th of February 2024, and I would not be surprised if people have proposed similar work from the end of 2023 to future. Also, I have only tested a limited number of baseline methods with a single dataset. Please make sure to read other related work around Mamba/Visual Mamba with UNet/VGG/ etc.

3. Q: Colloboration: I would like to discuss with other topic, like Image Registration, Human Pose Estimation, Image Fusion.

A: I would also like to do some work amazing. Connect with me via ziyang [dot] wang17 [at] gmail [dot] com

## Reference
```
@article{wang2024mamba,
  title={Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation},
  author={Wang, Ziyang and others},
  journal={arXiv preprint arXiv:2402.05079},
  year={2024}
}

```


## Acknowledgement
SSL4MIS, Segmamba, SwinUNet