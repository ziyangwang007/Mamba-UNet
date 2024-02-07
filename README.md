# Mamba-UNet

Mamba-UNet: Unet-like Pure Visual Mamba for Medical Image Segmentation

<img src="intro.png">
<img src="framework.png">


## Mamba-UNet Zoo

Supervised Mamba-UNet -> [[Paper Link]](https://github.com/ziyangwang007/Mamba-UNet/blob/main/MambaUNet.pdf) Released in 6/Feb/2024.

Semi-Supervised Mamba-UNet -> TBA

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
python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/unet --model unet
```

5. Train SwinUNet
```
python train_fully_supervised_2D_ViT.py --root_path ../data/ACDC --exp ACDC/swinunet --model swinunet
```

6. Train Mamba-UNet
```
python train_fully_supervised_2D_VIM.py --root_path ../data/ACDC --exp ACDC/VIM --model VIM
```

7. Test
```
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/xxx --model xxx
```




## Acknowledgement
SSL4MIS, Segmamba, SwinUNet