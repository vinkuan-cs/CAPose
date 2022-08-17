# 在M5上加如cross
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math
import os
import logging


from .hr_base import HRNET_base

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNET_base(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(HRNET_base, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, multi_scale_output=True)

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # print(len(y_list))        # 2
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # print(len(x_list))          # 3
        y_list = self.stage3(x_list)
        # print(len(y_list))        # 1
        return y_list

    def init_weights(self, pretrained='', print_load_info=False):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers and name in self.state_dict()\
                   or self.pretrained_layers[0] is '*':
                    existing_state_dict[name] = m
                    if print_load_info:
                        print(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:17, ...]).reshape(B, 17, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 17, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_dim=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:17, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AlternateAttention(nn.Module):
    def __init__(self, dim, num_token, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_tokens = num_token
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_l, x_s = x.split(self.num_tokens, dim=1)
        B, N_1, C = x_l.shape
        _, N_2, _ = x_s.shape
        q = self.wq(x_s).reshape(B, N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BN_2C -> BN_2H(C/H) -> BHN_2(C/H)
        k = self.wk(x_l).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BHN_1(C/H)
        v = self.wv(x_l).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN_2(C/H) @ BH(C/H)N_1 -> BHN_2N_1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)   # (BHN_2N_1 @ BHN_1(C/H)) -> BHN_2(C/H) -> BN_2H(C/H) -> BN_2C
        x = self.to_out(x)
        x = self.proj_drop(x)
        return x               # Alternate

class AlternateAttentionBlock(nn.Module):
    def __init__(self, dim, depth,  mlp_dim, num_heads, num_token = [],all_attn=[False], qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 dropout=0.):
        super().__init__()
        self.turns = depth
        self.layers = nn.ModuleList()
        self.num_token = num_token
        self.all_attn = all_attn
        for j in range(self.turns):
            for i in range(len(self.num_token)):
                if i < len(self.num_token) -1:
                    self.layers.append(nn.ModuleList([
                        PreNorm(dim, AlternateAttention(
                            dim, [self.num_token[i] + 17, self.num_token[(i + 1)] + 17], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                            proj_drop=drop)),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                    ]))
                else:
                    self.layers.append(nn.ModuleList([
                        PreNorm(dim, AlternateAttention(
                            dim, [self.num_token[0] + 17, self.num_token[i] + 17], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                            proj_drop=drop)),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                    ]))

        # print(len(self.layers))
    def forward(self, x, pos=None):
        assert len(x) == len(self.num_token), 'the length of x is wrong'

        for d in range(len(self.layers)//len(self.num_token)):
            for i in range(len(self.num_token)):
                if i < len(self.num_token) -1:
                    if pos[i] is not None:
                        x[i][:,17:] = x[i][:,17:] + pos[i]

                    if pos[i + 1] is not None:
                        x[i + 1][:,17:] = x[i +1][:,17:] + pos[i + 1]

                    y = torch.cat((x[i], x[i + 1]), dim=1)

                else:
                    if pos[0] is not None:
                        x[0][:, 17:] = x[0][:, 17:] + pos[0]

                    if pos[i] is not None:
                        x[i][:,17:] = x[i][:,17:] + pos[i]

                    y = torch.cat((x[0], x[i]), dim=1)

                if i < len(self.num_token) -2:
                    x[i + 1] = self.layers[d*len(self.num_token) + i][0](y) + x[i + 1]
                    x[i + 1] = self.layers[d*len(self.num_token) + i][1](x[i + 1])

                    k2, x[i + 1] = x[i + 1].split([17, self.num_token[i + 1]], dim=1)
                    k1, x[i] = x[i].split([17,self.num_token[i]], dim=1)

                    x[i + 1] = torch.cat((k1, x[i + 1]), dim=1)
                    x[i + 2] = torch.cat((k2, x[i + 2]), dim=1)


                elif i == len(self.num_token) - 2 :
                    x[i + 1] = self.layers[d * len(self.num_token) + i][0](y) + x[i + 1]
                    x[i + 1] = self.layers[d*len(self.num_token) + i][1](x[i + 1])

                    k1, x[i] = x[i].split([17, self.num_token[i]], dim=1)
                    x[0] = torch.cat((k1, x[0]), dim=1)

                else:
                    x[i] = self.layers[d * len(self.num_token) + i][0](y) + x[i]
                    x[i] = self.layers[d*len(self.num_token) + i][1](x[i])

                    k2, x[i] = x[i].split([17, self.num_token[i]], dim=1)
                    x[1] = torch.cat((k2, x[1]), dim=1)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                                scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None, pos=None):
        # print(x.shape[1])
        for idx, (attn, ff) in enumerate(self.layers):
            if self.all_attn:
                x[:, self.num_keypoints:] += pos
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class SenTrans(nn.Module):
    def __init__(self, *, branch, feature_size, patch_size, num_keypoints, dim, depth, scale_depth, heads, mlp_dim, apply_init=False,
                 hidden_heatmap_dim=64 * 6, heatmap_dim=64 * 48, heatmap_size=[64, 48], channels=[3], dropout=0.,
                 emb_dropout=0., pos_embedding_type=None, rever=False):
        super().__init__()

        assert isinstance(feature_size, list) and isinstance(patch_size, list), 'image_size and patch_size should be list'
        assert len(feature_size) == branch * 2, 'feature size is superfluous or insufficient'
        assert len(channels) == branch, 'channel is superfluous or insufficient'

        if rever:
            feature_size = list(reversed([feature_size[slice(i, i + 2)] for i in range(0, len(feature_size), 2)]))
            patch_size = list(reversed([patch_size[slice(i, i + 2)] for i in range(0, len(patch_size), 2)]))
            feature_size = sum(feature_size, [])
            patch_size = sum(patch_size, [])
            pos_embedding_type = list(reversed(pos_embedding_type))
            channels = list(reversed(channels))

        for i, j in enumerate(range(0, branch*2, 2)):
            assert feature_size[j] // (patch_size[j]) and feature_size[j+1] // (patch_size[j+1]), 'Image dimensions must be divisible by the patch size.'
            assert pos_embedding_type[i] in ['sine', 'learnable', 'sine-full']
        # b1,b2,b3: (h, w) that every branch yields

        num_patches = [(feature_size[i]//patch_size[i]) * (feature_size[i+1]// patch_size[i+1]) for i in range(0, branch* 2, 2)]
        # print('--------num_patches',num_patches)
        b = [(feature_size[i]//(patch_size[i]), feature_size[i+1]//patch_size[i+1]) for i in range(0, branch*2, 2)]

        patch_dim = [channels[j] * patch_size[i] * patch_size[i+1] for j, i in enumerate(range(0, branch*2, 2))]
        # print('------patch_dim', patch_dim)

        self.branch = branch
        self.inplanes = 64
        self.patch_size = patch_size

        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        # print('num_patches', num_patches)

        self.pos_embedding_type = pos_embedding_type
        self.rever = rever
        self.all_attn = [self.pos_embedding_type[i] == "sine-full" for i in range(len(self.pos_embedding_type))]


        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))

        if self.rever:
            self.res = nn.Conv2d(in_channels=channels[-1], out_channels=self.num_keypoints, kernel_size=1)
        else:
            self.res = nn.Conv2d(in_channels=channels[0], out_channels=self.num_keypoints, kernel_size=1)
        self.flat = nn.Flatten(-2,-1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(heatmap_dim),
            nn.Linear(heatmap_dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, dim))

        self.patch_to_embedding = nn.ModuleList([nn.Linear(patch_dim[i], dim) for i in range(self.branch)])

        self.pos_ebd = [self.make_position_embedding(b[i][1], b[i][0], dim,
                                                     pe_type=self.pos_embedding_type[i],
                                                     num_token=num_patches[i] + self.num_keypoints) for i in
                        range(0, self.branch)]
        self.pos = nn.Parameter(torch.cat(self.pos_ebd, dim=1))


        self.dropout = nn.Dropout(emb_dropout)

        self.stage = nn.ModuleList()
        self.crossfuse = nn.ModuleList()
        self.keypointfuse = nn.ModuleList()
        assert len(depth) == len(scale_depth), 'depths are not match each other'
        for i in range(len(depth)):
            sin_sta = nn.ModuleList()
            sin_sta.append(AttentionBlock(dim, depth[i], heads, mlp_dim, dropout, num_keypoints=num_keypoints,all_attn=self.all_attn[i], scale_with_head=True))
            sin_sta.append(AlternateAttentionBlock(dim, scale_depth[i],mlp_dim =mlp_dim, num_heads=heads, num_token=self.num_patches,all_attn=[self.all_attn[i % 3], self.all_attn[(i+1)%3]], qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.))
            self.stage.append(sin_sta)

            singlestage = nn.ModuleList()
            for j in range(self.branch):
                singlestage.append(CrossAttentionBlock(dim, heads, mlp_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), has_mlp=True))
            self.crossfuse.append(singlestage)

            self.keypointfuse.append(nn.Sequential(nn.LayerNorm(dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)))


        self.to_keypoint_token = nn.Identity()


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * len(self.stage)),
            nn.Linear(dim * len(self.stage), hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim * len(self.stage) <= hidden_heatmap_dim * 0.5) else nn.Sequential(
            nn.LayerNorm(dim * len(self.stage)),
            nn.Linear(dim * len(self.stage), heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def make_position_embedding(self, w, h, d_model, pe_type='sine', num_token=None):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            pos_embedding = None
            print("==> Without any PositionEmbedding~")
            return pos_embedding
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                pos_embedding = nn.Parameter(torch.zeros(1, num_token, d_model))
                trunc_normal_(pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
                return pos_embedding
            else:
                pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine or sine-full PositionEmbedding~")
                return pos_embedding

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear) or isinstance(m,nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def preforward(self,feature):
        out = self.res(feature)
        out = self.flat(out)
        out = self.mlp(out)
        return out

    def forward(self, feature, mask=None):
        res = self.preforward(feature[0])

        if self.rever:
            feature = list(reversed(feature))
        if isinstance(feature, list):
            b = feature[0].shape[0]
        else:
            b = feature.shape[0]
        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
        keypoint_tokens = res + keypoint_tokens

        # transformer
        x_set = []
        pos_attn = []
        per = 0
        p = self.patch_size
        for i in range(0, len(feature)):
            x = rearrange(feature[i], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p[2*i], p2=p[2*i+1])
            x = self.patch_to_embedding[i](x)
            n = x.shape[1]
            if i < 2:
                if self.pos_embedding_type[i] in ["sine"]:
                    #  "sine", ( keypoint)
                    x = x + self.pos[:, :n]
                    x = torch.cat((keypoint_tokens, x), dim=1)
                    pos_attn.append(None)
                    per = per + n
                elif self.pos_embedding_type[i] in ["sine-full"]:
                    # in the transformer encoder, add pos information
                    pos_attn.append(self.pos[:, :n])
                    x = torch.cat((keypoint_tokens, x), dim=1)
                    per = per + n
                else:
                    # 'learnable'
                    pos_attn.append(None)
                    x = torch.cat((keypoint_tokens, x), dim=1)
                    x = x + self.pos[:, :n + self.num_keypoints]
                    per = per + (n + self.num_keypoints)
            else:
                if self.pos_embedding_type[i] in ["sine", 'learnable']:
                    x = x + self.pos[:, per: per + n]
                    pos_attn.append(None)

                    per = per + (n + self.num_keypoints)
                else:
                    # in the transformer encoder, add pos information
                    pos_attn.append(self.pos[:, :n])
                    per = n + per

            x = self.dropout(x)

            x_set.append(x)

        k_all = []
        for i, blocks in enumerate(self.stage):
            x = blocks[0](x_set[0], mask=mask, pos=pos_attn[0])

            x_set[0] = x
            x_set = blocks[1](x_set, pos=pos_attn)

            # the second group of keypoint tokens is all in the x[1]
            k1, tmp1 = x_set[0].split([self.num_keypoints, self.num_patches[0]], dim=1)
            k2, tmp2 = x_set[1].split(
                [self.num_keypoints, self.num_patches[1]], dim=1)

            k1 = self.crossfuse[i][0](torch.cat((k1, x_set[-1]), dim=1))
            k2 = self.crossfuse[i][1](torch.cat((k2, tmp1), dim=1))
            k_all.append(self.keypointfuse[i](torch.cat((k1, k2), dim=-1)))

            x_set[0] = torch.cat((k1, tmp1), dim=1)
            x_set[1] = torch.cat((k2, tmp2), dim=1)

        x_out = []
        for i in range(len(k_all)):
            x_out.append(self.to_keypoint_token(k_all[i]))


        x = torch.cat(x_out, dim=2)

        # print(x.shape)              # torch.Size([1, 17, 576])
        x = self.mlp_head(x)

        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x



class CAPose(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA

        super(CAPose, self).__init__()

        ##################################################
        self.pre_feature = HRNET_base(cfg,**kwargs)
        self.transformer = SenTrans(branch=3, feature_size=[cfg.MODEL.IMAGE_SIZE[1]//4,cfg.MODEL.IMAGE_SIZE[0]//4,cfg.MODEL.IMAGE_SIZE[1]//8,cfg.MODEL.IMAGE_SIZE[0]//8,cfg.MODEL.IMAGE_SIZE[1]//16,cfg.MODEL.IMAGE_SIZE[0]//16],
                                    patch_size=[4, 3, 4, 3, 4, 3],
                            num_keypoints = cfg.MODEL.NUM_JOINTS,dim =cfg.MODEL.DIM,
                            channels=extra.STAGE3.NUM_CHANNELS,
                            depth=cfg.MODEL.TRANSFORMER_DEPTH, scale_depth=cfg.MODEL.MULTISCALE_TRANSFORMER_DEPTH,
                            heads=cfg.MODEL.TRANSFORMER_HEADS,
                            mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
                            apply_init=cfg.MODEL.INIT,
                            hidden_heatmap_dim=cfg.MODEL.HIDDEN_HEATMAP_DIM,
                            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
                            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
                            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
                            rever=False)
        ###################################################3

    def forward(self, x):
        x = self.pre_feature(x)

        x = self.transformer(x)

        return x

    def init_weights(self, pretrained='', cfg=None):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = CAPose(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED,cfg)
    return model


