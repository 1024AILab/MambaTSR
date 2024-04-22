# coding=utf-8
# @FileName:ConvNet.py
# @Time:2024/2/21 
# @Author: CZH
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from functools import reduce
import torch.nn.functional as F
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

"""
"""
class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)

class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

class SEAttention(nn.Module):
    def __init__(self, channel=3, reduction=3):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()

        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)

        )
        self.se = SEAttention()

        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        chaneel_1_max_pool = self.maxpool(x)
        desired_size = (x.size(2), x.size(3))
        channel_1_max_pool_out = F.interpolate(chaneel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        channel_2_1 = self.conv4(x)
        # print("channel_2_1", channel_2_1.shape)
        channel_2_2 = self.conv5(x)
        # print("channel_2_2", channel_2_2.shape)
        channel_2_3 = self.conv6(x)
        # print("channel_2_3", channel_2_3.shape)

        channel_2_sum = 0.2 * channel_2_1 + 0.6 * channel_2_2 + 0.2 * channel_2_3


        channel_2_x_1 = self.conv1(x)
        channel_2_x_2 = self.conv2(channel_2_x_1)
        channel_2_x_3 = self.conv3(channel_2_x_2)
        channel_2_x_4 = self.se(channel_2_x_3)

        channel_2_total = channel_2_x_4 * 0.6 + channel_2_x_3 * 0.4
        channel_2_total_avg_pool = self.avgpool(channel_2_total)
        desired_size = (x.size(2), x.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_total_avg_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        channel_3 = x
        # print("channel_1_max_pool_out", channel_1_max_pool_out.shape)
        # print("channel_2_avg_pool_out", channel_2_avg_pool_out.shape)
        # print("channel_3", channel_3.shape)
        total_3_channel = 0 * channel_1_max_pool_out + 1 * channel_2_avg_pool_out + 0 * channel_3
        return total_3_channel