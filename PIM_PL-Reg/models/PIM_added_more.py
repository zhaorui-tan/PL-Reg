import torch
import torch.nn as nn

import torch.nn.functional as F
from collections import OrderedDict




class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            elif len(shape) == 2:
                # CLIP-ViT: [B, C]
                b_shape = (1, shape[1])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps



class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features//2, num_features//2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features//2, num_features//2)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features//2, num_features//2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features//2, num_features//2)),
        ]))

        self.down1 = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features, num_features//2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features//2, num_features//2)),
        ]))
        self.down2 = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features, num_features // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features // 2, num_features // 2)),
        ]))

        self.up = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features// 2, num_features )),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        # x [batch, channel, w, h]
        # y [batch, emb_dim]
        x = self.down1(x)
        y = self.down1(y)
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.expand(size)
        bias = bias.expand(size)

        x = weight * x + bias
        x = self.up(x)
        return x



class PIM_partitioner(nn.Module):
    def __init__(self, num_features=512, num_classes=100, temp=25):
        super().__init__()

        # self.map = nn.Linear(num_features, num_features, bias=False)
        # self.y_map = nn.Linear(num_classes, num_features, bias=False)
        # self.filter = nn.Sequential(OrderedDict([nn.Linear(num_features, num_classes, bias=False),
        #
        #                                          ]))
        self.partitioner = nn.Linear(num_features, num_classes, bias=False)
        # nn.init.constant_(self.map.weight.data, 1/num_classes)
        self.temp = temp
        self.dropout = nn.Dropout(p=0.5)
        self.affine = affine(num_features)

        # self.mean_encoder = MeanEncoder(shape=[1, num_features * 2])

        # self.x_var_encoder = VarianceEncoder(shape=[1, num_features])
        # self.x_mean_encoder = MeanEncoder(shape=[1, num_features])
        # self.x_y_var_encoder = VarianceEncoder(shape=[1, num_features * 2])

    def forward(self, x, y=None, mb_lab_mask=None):

        # if y is not None:
        #     latent_x = self.map(x)
        #     latent_y = self.y_map(y)
        #
        #     x_mean =  self.x_mean_encoder(latent_x[mb_lab_mask])
        #     x_var = self.x_var_encoder(latent_x[mb_lab_mask])
        #     # vlb = (x_mean - x[mb_lab_mask]).pow(2).div(x_var) + x_var.log()
        #     vlb = x_var.log()
        #     x_reg_loss = vlb.mean() / 2.
        #
        #
        #     f = torch.cat([latent_x[mb_lab_mask], latent_y.detach()], dim=-1)
        #     # mean = mean_enc(f)
        #     var = self.x_y_var_encoder(f)
        #     reg_loss = var.log().mean() / 2.
        #
        #     feat_norm = l2normalize(latent_x[mb_lab_mask])
        #     y_norm = l2normalize(latent_y)
        #     x_shift = get_cond_shift(feat_norm, y_norm.detach(), )
        #
        #     y_out = self.partitioner(latent_y)
        #     y_out = y_out * self.temp
        #
        #
        #     out = self.partitioner(latent_x)
        #     out = out * self.temp
        #
        #     reg_loss_dict = {
        #         'x_reg_loss': x_reg_loss.item(),
        #         'reg_loss': reg_loss.item(),
        #         'x_shift': x_shift.item(),
        #     }
        #     return out, y_out, x_reg_loss + reg_loss, reg_loss_dict
        # else:
        # x = self.map(x)
        # x = self.dropout(x)
        # out1 = self.affine(x, x)
        out1 = x
        out = self.partitioner(out1)
        out = out * self.temp
        # out = F.relu(out)
        # return out, out1
        return out