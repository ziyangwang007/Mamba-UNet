from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.nnunet import initialize_network
from networks.unetr import UNETR
from monai.networks.nets import SwinUNETR


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "unetr":
        net = UNETR(
                    in_channels=in_chns,
                    out_channels=class_num,
                    img_size=(96, 96, 96),
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed='perceptron',
                    norm_name='instance',
                    conv_block=True,
                    res_block=True,
                    dropout_rate=0.0).cuda()
    elif net_type == "swinunetr":
        net = SwinUNETR(img_size=(64, 64, 64), in_channels=in_chns, out_channels=class_num, feature_size=48)
    else:
        net = None
    return net
