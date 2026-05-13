from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch.nn.functional as F
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
import torch.nn as nn
from src.model.model.anysplat import AnySplat
import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import random
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor
import torch.distributed as dist
import torch

from src.model.model.anysplat import AnySplat
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


class CrossAttentionFusion(nn.Module):
    def __init__(self, out_channels=512, seq_len=4096):
        super().__init__()

        # Step 1: Reduce feature_a from [1, 4, 81, 1029, 2048] to [1, N, 4096]
        self.feature_a_conv = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=1),               # reduce channels
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 4096)),           # -> [1, 8, 1, 1, 4096]
        )
        
        self.reg_weight = 0.05
        
        self.flatten = nn.Flatten(start_dim=1, end_dim=3) # -> [1, 8, 4096]

        # 1D conv to project feature_b from 8 -> 512
        self.zero_conv = nn.Conv1d(
            in_channels=8,
            out_channels=512,
            kernel_size=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm1d(512)  # normalize across channel dim
        # Initialize weights to zero
        nn.init.constant_(self.zero_conv.weight, 0.0)

    def forward(self, feature_b,feature_a):
        # feature_a: [1, 4, 81, 1029, 2048]
        # feature_b: [1, 512, 4096]

        # Step 1: Process feature_a to [1, 8, 4096]
        x_a = self.feature_a_conv(feature_a)  # [1, 8, 1, 1, 4096]
        x_a = self.flatten(x_a)               # [1, 8, 4096]
        # print("_a:", x_a.shape)
        # print("feature_a:", feature_a.shape)
        # print("feature_b:", feature_b.shape)

        # x_a = x_a.transpose(1, 2)  # (B, 4096, 8)
        a_proj = self.zero_conv(x_a)     # (B, 4096, 512)
        a_proj = self.bn(a_proj) 
        # a_proj = a_proj.transpose(1, 2)  # (B, 512, 4096)
        
        # Step 2: Concat with feature_b → [1, 520, 4096]
        fused = feature_b + a_proj

        mean_b = feature_b.mean(dim=[1, 2], keepdim=True)
        std_b = feature_b.std(dim=[1, 2], keepdim=True) + 1e-6

        mean_f = fused.mean(dim=[1, 2], keepdim=True)
        std_f = fused.std(dim=[1, 2], keepdim=True) + 1e-6

        reg_loss = self.reg_weight * (
            (mean_f - mean_b).pow(2).mean() + (std_f - std_b).pow(2).mean()
        )

        return fused, reg_loss

class To3D(nn.Module):
    def __init__(self, up_h=1029, up_w=2048,in_channels=1, mid_channels=8, out_channels=4):
        super().__init__()
        self.up_h = up_h
        self.up_w = up_w

        # Channel projection per input channel (grouped conv)
        self.channel_proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)  # consistent activation
            )

        # ---- Depth merging (depthwise conv) ----
        self.depth_merge = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=(16,3,3),
                padding=(0,1,1),
                bias=False
            ),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)  # consistent activation
        )
        # ---- Hierarchical spatial upsampling ----

        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 60->120
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 120->240
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 240->480
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 480->960
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, kernel_size=(70, 385), stride=1, padding=0),  # 960->1029x2048
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
        ])

        # ---- Final 1x1 projection per channel ----
        self.upsample = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True)  # changed from GELU to ReLU for consistency
        )


    def forward(self, x):
        B, d2, D, H, W = x.shape
        d1 = 4 * d2

        # --- Step 1: Channel projection ---
        x = x.view(B*d2, 1, D, H, W)            # treat each channel independently
        y = self.channel_proj(x)                 # [B*d2, 4, D, H, W]
        y = y.view(B, d1, D, H, W)              # [B, d1, D, H, W]

        # --- Step 2: Depth merge ---
        y = y.view(B*d1, 1, D, H, W)
        y = self.depth_merge(y)
        y = y.squeeze(2)                         # [B*d1, H, W]
        y = y.view(B, d1, H, W)                  # [B, d1, H, W]

        # --- Step 3: Hierarchical learnable upsampling ---
        y = y.view(B*d1, 1, H, W)               # grouped conv trick
        for layer in self.up_layers:
            y = layer(y)
        y = y.view(B, d1, self.up_h, self.up_w)  # [B, d1, 1029, 2048]

        # --- Step 4: Final projection ---
        #[rank2]: RuntimeError: Given groups=1, weight of size [4, 1, 1, 1, 1], expected input[1, 12, 1, 1029, 2048] to have 1 channels, but got 12 channels instead
        y = y.view(B*d1, 1, self.up_h, self.up_w)  # [B*d1, 1, H_up, W_up]
        y = self.upsample(y)                        # [B*d1, 4, H_up, W_up]
        y = torch.sigmoid(y) 
        y = y.view(B, d1, 4, self.up_h, self.up_w).permute(0, 2, 1, 3, 4)  # [B, 4, d1, H, W]
        return y

class To3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Step 1: [1,4,81,1029,2048] -> [1,1,81,1029,2048]
        self.proj1 = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        
        # Step 2: [1,1,81,1029,2048] -> [1,1,21,1029,2048]
        self.downsample_depth = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3,1,1), stride=(4,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

        # Step 4: [1,21,1029,2048] -> [1,21,240,2048]
        self.reduce_spatial_h = nn.Sequential(
            nn.Conv2d(1029, 240, kernel_size=1),
            nn.BatchNorm2d(240),
            nn.ReLU(inplace=True)
        )

        # Step 5: [1,21,240,2048] -> [1,21,240,416]
        self.reduce_spatial_w = nn.Sequential(
            nn.Conv2d(2048, 416, kernel_size=1),
            nn.BatchNorm2d(416),
            nn.ReLU(inplace=True)
        )
        
        self.zero_conv = nn.Conv3d(21, 21, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.zero_conv.weight)
        if self.zero_conv.bias is not None:
            nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        # x: [B,4,81,1029,2048]
        x = self.proj1(x)  # [B,1,81,1029,2048]
        x = self.downsample_depth(x)  # [B,1,21,1029,2048]
        x = x.squeeze(1)  # [B,21,1029,2048]

        # project spatial dims
        x = self.reduce_spatial_h(x.permute(0,2,1,3)).permute(0,2,1,3)  # [B,21,240,2048]
        x = self.reduce_spatial_w(x.permute(0,3,2,1)).permute(0,3,2,1)  # [B,21,240,416]

        # reshape to [1,21,16,60,104]
        x = x.reshape(x.size(0), x.size(1), 16, 60, 104)
        return x, self.zero_conv(x)


class To3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Step 1: [1,4,81,1029,2048] -> [1,1,81,1029,2048]
        # self.proj1 = nn.Sequential(
        #     nn.Conv3d(4, 1, kernel_size=1),
        #     nn.BatchNorm3d(1),
        #     nn.ReLU(inplace=True)
        # )
        
        # Step 2: [1,1,81,1029,2048] -> [1,1,21,1029,2048]
        # self.downsample_depth = nn.Sequential(
        #     nn.Conv3d(1, 1, kernel_size=(3,1,1), stride=(4,1,1), padding=(1,0,0)),
        #     nn.BatchNorm3d(1),
        #     nn.ReLU(inplace=True)
        # )

        # Step 4: [1,21,1029,2048] -> [1,21,240,2048]
        self.reduce_spatial_h = nn.Sequential(
            nn.Conv2d(1029, 240, kernel_size=1),
            nn.BatchNorm2d(240),
            nn.ReLU(inplace=True)
        )

        # Step 5: [1,21,240,2048] -> [1,21,240,416]
        self.reduce_spatial_w = nn.Sequential(
            nn.Conv2d(1024, 416, kernel_size=1),
            nn.BatchNorm2d(416),
            nn.ReLU(inplace=True)
        )
        
        self.zero_conv = nn.Conv2d(16, 16, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        if self.zero_conv.bias is not None:
            nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        # x: [B,4,81,1029,2048]
        # x = self.proj1(x)  # [B,1,81,1029,2048]
        # x = self.downsample_depth(x)  # [B,1,21,1029,2048]
        x = x.squeeze(1)  # [B,21,1029,2048]

        # project spatial dims
        x = self.reduce_spatial_h(x.permute(0,2,1,3)).permute(0,2,1,3)  # [B,21,240,2048]
        x = self.reduce_spatial_w(x.permute(0,3,2,1)).permute(0,3,2,1)  # [B,21,240,416]
        x = x.reshape(x.size(0), x.size(1), 16, 60*104)
        x_bias = self.zero_conv(x.permute(0,2,1,3)).permute(0,2,1,3)  # [B,21,240,416]
        x_bias = x_bias.reshape(x.size(0), x.size(1), 16, 60,104)
        # reshape to [1,21,16,60,104]
        # x = x.reshape(x.size(0), x.size(1), 16, 60, 104)
        return x, x_bias
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, out_channels=512, seq_len=4096):
#         super().__init__()

#         # Step 1: Reduce feature_a from [1, 4, 81, 1029, 2048] to [1, N, 4096]
#         self.feature_a_conv = nn.Sequential(
#             nn.Conv3d(4, 8, kernel_size=1),               # reduce channels
#             nn.BatchNorm3d(8),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d((1, 1, 4096)),           # -> [1, 8, 1, 1, 4096]
#         )
        
#         self.reg_weight = 0.05
        
#         self.flatten = nn.Flatten(start_dim=1, end_dim=3) # -> [1, 8, 4096]

#         # 1D conv to project feature_b from 8 -> 512
#         self.zero_conv = nn.Conv1d(
#             in_channels=8,
#             out_channels=512,
#             kernel_size=1,
#             bias=False
#         )
        
#         self.bn = nn.BatchNorm1d(512)  # normalize across channel dim
#         # Initialize weights to zero
#         nn.init.constant_(self.zero_conv.weight, 0.0)

#     def forward(self, feature_b,feature_a):
#         # feature_a: [1, 4, 81, 1029, 2048]
#         # feature_b: [1, 512, 4096]

#         # Step 1: Process feature_a to [1, 8, 4096]
#         x_a = self.feature_a_conv(feature_a)  # [1, 8, 1, 1, 4096]
#         x_a = self.flatten(x_a)               # [1, 8, 4096]
#         # print("_a:", x_a.shape)
#         # print("feature_a:", feature_a.shape)
#         # print("feature_b:", feature_b.shape)

#         # x_a = x_a.transpose(1, 2)  # (B, 4096, 8)
#         a_proj = self.zero_conv(x_a)     # (B, 4096, 512)
#         a_proj = self.bn(a_proj) 
#         # a_proj = a_proj.transpose(1, 2)  # (B, 512, 4096)
        
#         # Step 2: Concat with feature_b → [1, 520, 4096]
#         fused = feature_b + a_proj

#         mean_b = feature_b.mean(dim=[1, 2], keepdim=True)
#         std_b = feature_b.std(dim=[1, 2], keepdim=True) + 1e-6

#         mean_f = fused.mean(dim=[1, 2], keepdim=True)
#         std_f = fused.std(dim=[1, 2], keepdim=True) + 1e-6

#         reg_loss = self.reg_weight * (
#             (mean_f - mean_b).pow(2).mean() + (std_f - std_b).pow(2).mean()
#         )

#         return fused, reg_loss

# class To3D(nn.Module):
#     def __init__(self, up_h=1029, up_w=2048,in_channels=1, mid_channels=8, out_channels=4):
#         super().__init__()
#         self.up_h = up_h
#         self.up_w = up_w

#         # Channel projection per input channel (grouped conv)
#         self.channel_proj = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm3d(out_channels),
#                 nn.ReLU(inplace=True)  # consistent activation
#             )

#         # ---- Depth merging (depthwise conv) ----
#         self.depth_merge = nn.Sequential(
#             nn.Conv3d(
#                 in_channels=1,
#                 out_channels=1,
#                 kernel_size=(16,3,3),
#                 padding=(0,1,1),
#                 bias=False
#             ),
#             nn.BatchNorm3d(1),
#             nn.ReLU(inplace=True)  # consistent activation
#         )
#         # ---- Hierarchical spatial upsampling ----

#         self.up_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 60->120
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 120->240
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 240->480
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 480->960
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose2d(1, 1, kernel_size=(70, 385), stride=1, padding=0),  # 960->1029x2048
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True)
#             )
#         ])

#         # ---- Final 1x1 projection per channel ----
#         self.upsample = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True)  # changed from GELU to ReLU for consistency
#         )


#     def forward(self, x):
#         B, d2, D, H, W = x.shape
#         d1 = 4 * d2

#         # --- Step 1: Channel projection ---
#         x = x.view(B*d2, 1, D, H, W)            # treat each channel independently
#         y = self.channel_proj(x)                 # [B*d2, 4, D, H, W]
#         y = y.view(B, d1, D, H, W)              # [B, d1, D, H, W]

#         # --- Step 2: Depth merge ---
#         y = y.view(B*d1, 1, D, H, W)
#         y = self.depth_merge(y)
#         y = y.squeeze(2)                         # [B*d1, H, W]
#         y = y.view(B, d1, H, W)                  # [B, d1, H, W]

#         # --- Step 3: Hierarchical learnable upsampling ---
#         y = y.view(B*d1, 1, H, W)               # grouped conv trick
#         for layer in self.up_layers:
#             y = layer(y)
#         y = y.view(B, d1, self.up_h, self.up_w)  # [B, d1, 1029, 2048]

#         # --- Step 4: Final projection ---
#         #[rank2]: RuntimeError: Given groups=1, weight of size [4, 1, 1, 1, 1], expected input[1, 12, 1, 1029, 2048] to have 1 channels, but got 12 channels instead
#         y = y.view(B*d1, 1, self.up_h, self.up_w)  # [B*d1, 1, H_up, W_up]
#         y = self.upsample(y)                        # [B*d1, 4, H_up, W_up]
#         y = y.view(B, d1, 4, self.up_h, self.up_w).permute(0, 2, 1, 3, 4)  # [B, 4, d1, H, W]
#         return y
    
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, out_channels=512, seq_len=4096):
#         super().__init__()

#         # Step 1: Reduce feature_a from [1, 4, 81, 1029, 2048] to [1, N, 4096]
#         self.feature_a_conv = nn.Sequential(
#             nn.Conv3d(4, 8, kernel_size=1),               # reduce channels
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d((1, 1, 4096)),           # -> [1, 8, 1, 1, 4096]
#         )
        
#         self.flatten = nn.Flatten(start_dim=1, end_dim=3) # -> [1, 8, 4096]

#         # 1D conv to project feature_b from 8 -> 512
#         self.zero_conv = nn.Conv1d(
#             in_channels=8,
#             out_channels=512,
#             kernel_size=1,
#             bias=False
#         )
#         # Initialize weights to zero
#         nn.init.constant_(self.zero_conv.weight, 0.0)

#     def forward(self, feature_b,feature_a):
#         # feature_a: [1, 4, 81, 1029, 2048]
#         # feature_b: [1, 512, 4096]

#         # Step 1: Process feature_a to [1, 8, 4096]
#         x_a = self.feature_a_conv(feature_a)  # [1, 8, 1, 1, 4096]
#         x_a = self.flatten(x_a)               # [1, 8, 4096]
#         # print("_a:", x_a.shape)
#         # print("feature_a:", feature_a.shape)
#         # print("feature_b:", feature_b.shape)

#         # x_a = x_a.transpose(1, 2)  # (B, 4096, 8)
#         a_proj = self.zero_conv(x_a)     # (B, 4096, 512)
#         # a_proj = a_proj.transpose(1, 2)  # (B, 512, 4096)
        
#         # Step 2: Concat with feature_b → [1, 520, 4096]
#         fused = feature_b + a_proj
#         return fused
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, embed_dim=256, num_heads=4):
#         super().__init__()

#         # ↓↓↓↓↓↓ Feature A Processing ↓↓↓↓↓↓
#         # Instead of mean over temporal dim (4), use Conv1d with kernel=4 to learn temporal fusion
#         self.temporal_conv_a = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        
#         self.reduce_a = nn.Sequential(
#             # Step 1: 2048 -> 512 channels, kernel_size=1
#             nn.Conv1d(2048, 512, kernel_size=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),

#             # Step 2: downsample width 1029 -> 343, channels stay 512
#             nn.Conv1d(512, 512, kernel_size=3, stride=3, padding=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),

#             # Step 3: 512 -> 256 channels, kernel_size=1
#             nn.Conv1d(512, 256, kernel_size=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#         )

#         # ↓↓↓↓↓↓ Feature B Processing ↓↓↓↓↓↓
#         self.reduce_b = nn.Sequential(
#             # Step 1: Reduce channels 104 -> 52, kernel_size=1 (no spatial change)
#             nn.Conv3d(104, 52, kernel_size=1),
#             nn.BatchNorm3d(52),
#             nn.ReLU(inplace=True),
            
#             # Step 2: Downsample spatial dims by 2 (height:16->8, width:60->30), keep depth (n)
#             nn.Conv3d(52, 52, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
#             nn.BatchNorm3d(52),
#             nn.ReLU(inplace=True),

#             # Step 3: Reduce channels 52 -> 26, kernel_size=1
#             nn.Conv3d(52, 26, kernel_size=1),
#             nn.BatchNorm3d(26),
#             nn.ReLU(inplace=True),

#             # Step 4: Downsample spatial dims by 2 again (height:8->4, width:30->15), keep depth
#             nn.Conv3d(26, 26, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
#             nn.BatchNorm3d(26),
#             nn.ReLU(inplace=True),
#         )

#         self.proj_b = nn.Sequential(
#             nn.Flatten(1),                            # [B*d2, 26, 4, 15] → [B*d2, 1560]
#             nn.Linear(26 * 4 * 15, embed_dim),        # → [B*d2, 256]
#             nn.ReLU(),
#             nn.LayerNorm(embed_dim)
#         )

#         # ↓↓↓↓↓↓ Attention ↓↓↓↓↓↓
#         self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

#         # ↓↓↓↓↓↓ Learnable upsampling replacing interpolate ↓↓↓↓↓↓
#         self.upsample_b = nn.Sequential(
#             nn.ConvTranspose3d(embed_dim, 128, kernel_size=(1,4,4), stride=(1,4,4)),  # [B, 256, d2, 1, 1] → [B, 128, d2, 4, 4]
#             nn.ReLU(),
#             nn.ConvTranspose3d(128, 104, kernel_size=(1,4,15), stride=(1,4,15)),      # [B, 128, d2, 4, 4] → [B, 104, d2, 16, 60]
#             nn.ReLU(),
#         )

#         self.out_conv = nn.Conv3d(104, 104, kernel_size=1)  # Final residual adjustment
#         nn.init.zeros_(self.out_conv.weight)
#         nn.init.zeros_(self.out_conv.bias)
        
#     def forward(self, feature_a, feature_b):
#         B, T1, d1, H1, C1 = feature_a.shape   # [B, 4, d1, 1029, 2048]
#         B, d2, H2, W2, C2 = feature_b.shape   # [B, d2, 16, 60, 104]

#         # Reshape A: [B, 4, d1, 1029, 2048] → [B*d1, 4, 1029, 2048]
#         feat_a = feature_a.permute(0, 2, 1, 3, 4).contiguous().reshape(B * d1, 4, 1029, 2048)

#         # Apply temporal conv over the temporal dim (4)
#         # First flatten last dim (2048) to treat as channels for conv1d over temporal dim
#         # Start from [B*d1, 4, 1029, 2048]
#         feat_a = feature_a.permute(0, 2, 1, 3, 4).contiguous().reshape(B * d1, 4, 1029, 2048)

#         # Move channel to second dim for Conv1d over temporal dim
#         feat_a = feat_a.permute(0, 3, 1, 2)  # [B*d1, 2048, 4, 1029]
#         feat_a = feat_a.reshape(B * d1 * 1029, 2048, 4).transpose(1, 2)  # [B*d1*1029, 4, 2048]
#         temp_fused = self.temporal_conv_a(feat_a)  # [B*d1*1029, 1, 1]

#         temp_fused = temp_fused.squeeze(1)          # [B*d1*1029, 2048]

#         # Now reshape to [B*d1, 1029, 2048]
#         temp_fused = temp_fused.reshape(B * d1, 1029, 2048).permute(0, 2, 1)  # [B*d1, 2048, 1029]
#         reduced_a = self.reduce_a(temp_fused)  # [B*d1, 256, 343]
#         reduced_a = reduced_a.permute(0, 2, 1).reshape(B, d1, 343, -1)  # [B, d1, 343, 256]

#         # Attention KV: flatten sequence
#         kv = reduced_a.reshape(B, d1 * 343, -1)  # [B, d1*343, 256]

#         # Reshape B: [B, d2, 16, 60, 104] → [B, 104, d2, 16, 60]
#         feat_b = feature_b.permute(0, 4, 1, 2, 3).contiguous()
#         reduced_b = self.reduce_b(feat_b)  # [B, 26, d2, 4, 15]
#         reduced_b = reduced_b.permute(0, 2, 1, 3, 4).contiguous()  # [B, d2, 26, 4, 15]
#         reduced_b = reduced_b.reshape(B * d2, 26, 4, 15)

#         q = self.proj_b(reduced_b).reshape(B, d2, -1)  # [B, d2, 256]

#         # Attention Q: [B, d2, 256]; KV: [B, d1*343, 256]
#         attn_out, _ = self.attn(q, kv, kv)  # → [B, d2, 256]

#         # Upsample back to [B, 104, d2, 16, 60] using learnable ConvTranspose3d
#         up = attn_out.reshape(B, d2, -1, 1, 1).permute(0, 2, 1, 3, 4)  # [B, 256, d2, 1, 1]
#         upsampled = self.upsample_b(up)  # [B, 104, d2, 16, 60]

#         # Match shape to feature_b
#         upsampled = upsampled.permute(0, 2, 3, 4, 1)  # [B, d2, 16, 60, 104]

#         # Final residual conv
#         delta = self.out_conv(upsampled.permute(0, 4, 1, 2, 3))  # [B, 104, d2, 16, 60]
#         delta = delta.permute(0, 2, 3, 4, 1)  # [B, d2, 16, 60, 104]

#         return feature_b + delta  # Residual fusion

# class To3D(nn.Module):
#     def __init__(self, up_h=1029, up_w=2048):
#         super().__init__()
#         self.up_h = up_h
#         self.up_w = up_w

#         # Channel projection per input channel (grouped conv)
#         self.channel_proj = nn.Conv3d(
#             in_channels=1, out_channels=4, kernel_size=1
#         )  # will be applied per channel dynamically

#         # Depth merging (depthwise conv)
#         self.depth_merge = nn.Conv3d(
#             in_channels=1, out_channels=1, kernel_size=(16,3,3), padding=(0,1,1)
#         )

#         # Learnable hierarchical spatial upsampling
#         # We keep ConvTranspose2d with 1 channel input and output per group
#         self.up_layers = nn.ModuleList([
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 60->120
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 120->240
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 240->480
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 480->960
#             nn.ConvTranspose2d(1, 1, kernel_size=(70,385), stride=1, padding=0)  # precise to 1029x2048
#         ])

#         # Final 1x1 projection per channel
#         self.upsample = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=1),
#             nn.BatchNorm2d(4),
#             nn.GELU()
#         )

#     def forward(self, x):
#         B, d2, D, H, W = x.shape
#         d1 = 4 * d2

#         # --- Step 1: Channel projection ---
#         x = x.view(B*d2, 1, D, H, W)            # treat each channel independently
#         y = self.channel_proj(x)                 # [B*d2, 4, D, H, W]
#         y = y.view(B, d1, D, H, W)              # [B, d1, D, H, W]

#         # --- Step 2: Depth merge ---
#         y = y.view(B*d1, 1, D, H, W)
#         y = self.depth_merge(y)
#         y = y.squeeze(2)                         # [B*d1, H, W]
#         y = y.view(B, d1, H, W)                  # [B, d1, H, W]

#         # --- Step 3: Hierarchical learnable upsampling ---
#         y = y.view(B*d1, 1, H, W)               # grouped conv trick
#         for layer in self.up_layers:
#             y = layer(y)
#             y = F.gelu(y)
#         y = y.view(B, d1, self.up_h, self.up_w)  # [B, d1, 1029, 2048]

#         # --- Step 4: Final projection ---
#         #[rank2]: RuntimeError: Given groups=1, weight of size [4, 1, 1, 1, 1], expected input[1, 12, 1, 1029, 2048] to have 1 channels, but got 12 channels instead
#         y = y.view(B*d1, 1, self.up_h, self.up_w)  # [B*d1, 1, H_up, W_up]
#         y = self.upsample(y)                        # [B*d1, 4, H_up, W_up]
#         y = y.view(B, d1, 4, self.up_h, self.up_w).permute(0, 2, 1, 3, 4)  # [B, 4, d1, H, W]
#         return y


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae
        self.fusion = CrossAttentionFusion()
        self.fusion.eval()
        
        self.proj = To3D()
        self.proj.eval()
        # self.fusion.requires_grad_(True)

        self.model_3D = AnySplat.from_pretrained("lhjiang/anysplat")
        self.model_3D.eval()
        
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def crop_img(self, img):
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)  # Convert tensor to PIL image

        width, height = img.size
        if width > height:
            new_height = 448
            new_width = int(width * (new_height / height))
        else:
            new_width = 448
            new_height = int(height * (new_width / width))
        img = img.resize((new_width, new_height))

        # Center crop
        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        right = left + 448
        bottom = top + 448
        img = img.crop((left, top, right, bottom))

        img_tensor = ToTensor()(img) * 2.0 - 1.0  # [-1, 1]
        return img_tensor
    
    def video_to_3D_feature(self, video):
        video = (video * 0.5 + 0.5).clamp(0, 1) #(1,81,3,480,832)
        video = rearrange(video, 'b t c h w -> b t h w c')
        cropped_imgs = [self.crop_img(rearrange(video, 'b t h w c -> b t c h w')[0,i]) for i in range(video.shape[1])]
        cropped_imgs = torch.stack(cropped_imgs, dim=0).unsqueeze(0) # [1, K, 3, 448, 448]
        b, v, _, h, w = cropped_imgs.shape
        feature_list = self.model.model_3D.inference((cropped_imgs+1)*0.5)
        fused_feature = torch.cat(feature_list)
        return fused_feature
    
    # def inference(
    #     self,
    #     noise: torch.Tensor,
    #     text_prompts: List[str],
    #     initial_latent: Optional[torch.Tensor] = None,
    #     return_latents: bool = False,
    #     profile: bool = False,
    #     low_memory: bool = False,
    # ) -> torch.Tensor:
    #     """
    #     Perform inference on the given noise and text prompts.
    #     Inputs:
    #         noise (torch.Tensor): The input noise tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #         text_prompts (List[str]): The list of text prompts.
    #         initial_latent (torch.Tensor): The initial latent tensor of shape
    #             (batch_size, num_input_frames, num_channels, height, width).
    #             If num_input_frames is 1, perform image to video.
    #             If num_input_frames is greater than 1, perform video extension.
    #         return_latents (bool): Whether to return the latents.
    #     Outputs:
    #         video (torch.Tensor): The generated video tensor of shape
    #             (batch_size, num_output_frames, num_channels, height, width).
    #             It is normalized to be in the range [0, 1].
    #     """
    #     batch_size, num_frames, num_channels, height, width = noise.shape
    #     if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
    #         # If the first frame is independent and the first frame is provided, then the number of frames in the
    #         # noise should still be a multiple of num_frame_per_block
    #         assert num_frames % self.num_frame_per_block == 0
    #         num_blocks = num_frames // self.num_frame_per_block
    #     else:
    #         # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
    #         assert (num_frames - 1) % self.num_frame_per_block == 0
    #         num_blocks = (num_frames - 1) // self.num_frame_per_block
    #     num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
    #     num_output_frames = num_frames + num_input_frames  # add the initial latent frames
    #     conditional_dict = self.text_encoder(
    #         text_prompts=text_prompts
    #     )

    #     if low_memory:
    #         gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
    #         move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

    #     output = torch.zeros(
    #         [batch_size, num_output_frames, num_channels, height, width],
    #         device=noise.device,
    #         dtype=noise.dtype
    #     )

    #     # Set up profiling if requested
    #     if profile:
    #         init_start = torch.cuda.Event(enable_timing=True)
    #         init_end = torch.cuda.Event(enable_timing=True)
    #         diffusion_start = torch.cuda.Event(enable_timing=True)
    #         diffusion_end = torch.cuda.Event(enable_timing=True)
    #         vae_start = torch.cuda.Event(enable_timing=True)
    #         vae_end = torch.cuda.Event(enable_timing=True)
    #         block_times = []
    #         block_start = torch.cuda.Event(enable_timing=True)
    #         block_end = torch.cuda.Event(enable_timing=True)
    #         init_start.record()

    #     # Step 1: Initialize KV cache to all zeros
    #     if self.kv_cache1 is None:
    #         self._initialize_kv_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #         self._initialize_crossattn_cache(
    #             batch_size=batch_size,
    #             dtype=noise.dtype,
    #             device=noise.device
    #         )
    #     else:
    #         # reset cross attn cache
    #         for block_index in range(self.num_transformer_blocks):
    #             self.crossattn_cache[block_index]["is_init"] = False
    #         # reset kv cache
    #         for block_index in range(len(self.kv_cache1)):
    #             self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)
    #             self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
    #                 [0], dtype=torch.long, device=noise.device)

    #     # Step 2: Cache context feature
    #     current_start_frame = 0
    #     if initial_latent is not None:
    #         timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
    #         if self.independent_first_frame:
    #             # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
    #             assert (num_input_frames - 1) % self.num_frame_per_block == 0
    #             num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
    #             output[:, :1] = initial_latent[:, :1]
    #             self.generator(
    #                 noisy_image_or_video=initial_latent[:, :1],
    #                 conditional_dict=conditional_dict,
    #                 timestep=timestep * 0,
    #                 kv_cache=self.kv_cache1,
    #                 crossattn_cache=self.crossattn_cache,
    #                 current_start=current_start_frame * self.frame_seq_length,
    #             )
    #             current_start_frame += 1
    #         else:
    #             # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
    #             assert num_input_frames % self.num_frame_per_block == 0
    #             num_input_blocks = num_input_frames // self.num_frame_per_block

    #         for _ in range(num_input_blocks):
    #             current_ref_latents = \
    #                 initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
    #             output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
    #             self.generator(
    #                 noisy_image_or_video=current_ref_latents,
    #                 conditional_dict=conditional_dict,
    #                 timestep=timestep * 0,
    #                 kv_cache=self.kv_cache1,
    #                 crossattn_cache=self.crossattn_cache,
    #                 current_start=current_start_frame * self.frame_seq_length,
    #             )
    #             current_start_frame += self.num_frame_per_block

    #     if profile:
    #         init_end.record()
    #         torch.cuda.synchronize()
    #         diffusion_start.record()

    #     # Step 3: Temporal denoising loop
    #     all_num_frames = [self.num_frame_per_block] * num_blocks
    #     if self.independent_first_frame and initial_latent is None:
    #         all_num_frames = [1] + all_num_frames
    #     for current_num_frames in all_num_frames:
    #         if profile:
    #             block_start.record()

    #         noisy_input = noise[
    #             :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

    #         # Step 3.1: Spatial denoising loop
    #         for index, current_timestep in enumerate(self.denoising_step_list):
    #             print(f"current_timestep: {current_timestep}")
    #             # set current timestep
    #             timestep = torch.ones(
    #                 [batch_size, current_num_frames],
    #                 device=noise.device,
    #                 dtype=torch.int64) * current_timestep

    #             if index < len(self.denoising_step_list) - 1:
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=conditional_dict,
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )
    #                 next_timestep = self.denoising_step_list[index + 1]
    #                 noisy_input = self.scheduler.add_noise(
    #                     denoised_pred.flatten(0, 1),
    #                     torch.randn_like(denoised_pred.flatten(0, 1)),
    #                     next_timestep * torch.ones(
    #                         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
    #                 ).unflatten(0, denoised_pred.shape[:2])
    #             else:
    #                 # for getting real output
    #                 _, denoised_pred = self.generator(
    #                     noisy_image_or_video=noisy_input,
    #                     conditional_dict=conditional_dict,
    #                     timestep=timestep,
    #                     kv_cache=self.kv_cache1,
    #                     crossattn_cache=self.crossattn_cache,
    #                     current_start=current_start_frame * self.frame_seq_length
    #                 )

    #         # Step 3.2: record the model's output
    #         output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

    #         # Step 3.3: rerun with timestep zero to update KV cache using clean context
    #         context_timestep = torch.ones_like(timestep) * self.args.context_noise
    #         self.generator(
    #             noisy_image_or_video=denoised_pred,
    #             conditional_dict=conditional_dict,
    #             timestep=context_timestep,
    #             kv_cache=self.kv_cache1,
    #             crossattn_cache=self.crossattn_cache,
    #             current_start=current_start_frame * self.frame_seq_length,
    #         )

    #         if profile:
    #             block_end.record()
    #             torch.cuda.synchronize()
    #             block_time = block_start.elapsed_time(block_end)
    #             block_times.append(block_time)

    #         # Step 3.4: update the start and end frame indices
    #         current_start_frame += current_num_frames

    #     if profile:
    #         # End diffusion timing and synchronize CUDA
    #         diffusion_end.record()
    #         torch.cuda.synchronize()
    #         diffusion_time = diffusion_start.elapsed_time(diffusion_end)
    #         init_time = init_start.elapsed_time(init_end)
    #         vae_start.record()

    #     # Step 4: Decode the output
    #     video = self.vae.decode_to_pixel(output, use_cache=False)
    #     video = (video * 0.5 + 0.5).clamp(0, 1)

    #     if profile:
    #         # End VAE timing and synchronize CUDA
    #         vae_end.record()
    #         torch.cuda.synchronize()
    #         vae_time = vae_start.elapsed_time(vae_end)
    #         total_time = init_time + diffusion_time + vae_time

    #         print("Profiling results:")
    #         print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
    #         print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
    #         for i, block_time in enumerate(block_times):
    #             print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
    #         print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
    #         print(f"  - Total time: {total_time:.2f} ms")

    #     # if return_latents:
    #     #     return video, output
    #     # else:
    #     #     return video
    #     return output

    def inference(
        self,
        noise: torch.Tensor,
        # text_prompts: List[str],
        conditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        # if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
        #     # If the first frame is independent and the first frame is provided, then the number of frames in the
        #     # noise should still be a multiple of num_frame_per_block
        #     assert num_frames % self.num_frame_per_block == 0
        #     num_blocks = num_frames // self.num_frame_per_block
        # else:
        #     # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
        #     assert (num_frames - 1) % self.num_frame_per_block == 0
        #     num_blocks = (num_frames - 1) // self.num_frame_per_block
        
        # if initial_latent is not None and initial_latent.shape[1] == 1:
        #     self.independent_first_frame = True
        # else:
        #     self.independent_first_frame = False
        
        # False
        # True, not None
        # True, None
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
            
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        # conditional_dict = self.text_encoder(
        #     text_prompts=text_prompts
        # )
        
        # print("###############describe conditional dict")
        # print(conditional_dict.keys())
        # if initial_latent is not None:
            # y_flat = initial_latent[:,0:1].view(1, -1)  # shape: (1, 16*60*104) = (1, 99840)

            # # Step 2: Interpolate to match final shape
            # y_resized = torch.nn.functional.interpolate(
            #     y_flat.unsqueeze(1), size=(512 * 4096), mode='linear', align_corners=False
            # )  # shape: (1, 1, 512*4096)

            # # Step 3: Reshape to (1, 512, 4096)
            # y_shaped = y_resized.view(1, 512, 4096)

            # # Combine
            # combined = conditional_dict["prompt_embeds"] + y_shaped
            
            # conditional_dict["image_latent"] = initial_latent[:,0:1] ###removed
            # conditional_dict["prompt_embeds"] = combined
            
            # dict_keys(['prompt_embeds'])
            # torch.Size([1, 512, 4096])
            # torch.Size([1, 3, 57, 480, 832])
                

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        
        # video = self.vae.decode_to_pixel(output, use_cache=False)
        # video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        # if return_latents:
        #     return video, output
        # else:
        #     return video
        return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
