# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torchvision.models
import clip


def clip_imageencoder(name):
    model, _preprocess = clip.load(name, device="cpu")
    imageencoder = model.visual

    return imageencoder


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchhub_load(repo, model, **kwargs):
    try:
        # torch >= 1.10
        try:
            network = torch.hub.load(repo, model=model, skip_validation=True, **kwargs)
        except FileNotFoundError:
        # torch 1.7.1
        # try:

            network = torch.hub.load(repo, model=model, skip_validation=True, source='local', **kwargs)
    except:
        # repo = "/Data_PHD_Backup/phd21_zhaorui_tan/current/torch_cache/hub/facebookresearch_swag_main"
        # try:
        #     repo = '/Data_PHD/phd22_zhaorui_tan/toolkit/torch_cache/hub/facebookresearch_swag_master'
        #     network = torch.hub.load(repo, model=model, skip_validation=True, source='local', **kwargs)
        # except:
        # repo = "/Data_PHD_Backup/phd21_zhaorui_tan/current/torch_cache/hub/facebookresearch_swag_main"
        repo = "/Data_PHD_Backup/phd21_zhaorui_tan/current/toolkits/torch_cache/hub/facebookresearch_swag_main"
        network = torch.hub.load(repo, model=model, skip_validation=True, source='local', **kwargs)
    return network


def get_backbone(name, preserve_readout, pretrained):
    # if not pretrained:
    #     assert name in ["resnet50", "swag_regnety_16gf"], "Only RN50/RegNet supports non-pretrained network"

    if name == "resnet18":
        network = torchvision.models.resnet18(pretrained=True)
        n_outputs = 512
    elif name == "resnet50":
        network = torchvision.models.resnet50(pretrained=True)
        n_outputs = 2048
    elif name == "resnet50_barlowtwins":
        network = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        n_outputs = 2048
    elif name == "resnet50_moco":
        network = torchvision.models.resnet50()

        # download pretrained model of MoCo v3: https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar
        ckpt_path = "./r-50-1000ep.pth.tar"

        # https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py#L172
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        linear_keyword = "fc"  # resnet linear keyword
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = network.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))

        n_outputs = 2048
    elif name.startswith("clip_resnet"):
        name = "RN" + name[11:]
        network = clip_imageencoder(name)
        n_outputs = network.output_dim
    elif name == "clip_vit-b16":
        network = clip_imageencoder("ViT-B/16")
        n_outputs = network.output_dim
    elif name == "swag_regnety_16gf":
        # No readout layer as default


        network = torchhub_load("facebookresearch/swag", model="regnety_16gf", pretrained=pretrained)

        network.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        n_outputs = 3024
    else:
        raise ValueError(name)

    if not preserve_readout:
        # remove readout layer (but left GAP and flatten)
        # final output shape: [B, n_outputs]
        if name.startswith("resnet"):
            del network.fc
            network.fc = Identity()

    return network, n_outputs
