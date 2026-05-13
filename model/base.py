from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch.nn.functional as F

from src.model.model.anysplat import AnySplat


# class CrossAttentionFusion(nn.Module):
#     def __init__(self, out_channels=512, latent_dim=4096, reg_weight=0.05):
#         """
#         Args:
#             out_channels: output channels (should match feature_b)
#             latent_dim: projection dimension
#             reg_weight: weight for distribution regularization
#         """
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.reg_weight = reg_weight

#         # Step 1. Feature reduction (spatial-temporal)
#         self.feature_a_conv = nn.Sequential(
#             nn.Conv3d(4, 16, kernel_size=1),
#             nn.BatchNorm3d(16),
#             nn.ReLU(inplace=True),
#         )

#         # Step 2. Adaptive pooling — preserve time dim
#         self.spatial_reduce = nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, 16, T, 1, 1]

#         # Step 3. Projection layers
#         self.temporal_proj = None  # initialized dynamically
#         self.channel_proj = nn.Conv1d(16, out_channels, kernel_size=1)
#         nn.init.zeros_(self.channel_proj.weight)

#         # Optional affine transformation for matching distribution
#         self.affine = nn.Parameter(torch.ones(1, out_channels, 1))
#         self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))

#     def forward(self, feature_b, feature_a):
#         """
#         Args:
#             feature_a: [B, 4, T, H, W]
#             feature_b: [B, C, latent_dim]
#         Returns:
#             fused: [B, C, latent_dim]
#             reg_loss: scalar tensor for distribution regularization
#         """
#         B, _, T, H, W = feature_a.shape
#         C, L = feature_b.shape[1], feature_b.shape[2]

#         # Step 1: spatial reduction
#         x_a = self.feature_a_conv(feature_a)       # [B, 16, T, H, W]
#         x_a = self.spatial_reduce(x_a).squeeze(-1).squeeze(-1)  # [B, 16, T]

#         # Step 2: dynamic temporal projection to latent_dim
#         if self.temporal_proj is None or self.temporal_proj.in_features != T:
#             self.temporal_proj = nn.Linear(T, L, bias=False).to(x_a.device)
#             nn.init.xavier_uniform_(self.temporal_proj.weight)

#         x_a = self.temporal_proj(x_a)              # [B, 16, L]
#         a_proj = self.channel_proj(x_a)            # [B, C, L]

#         # Step 3: Apply affine scaling to stabilize variance
#         a_proj = a_proj * self.affine + self.bias

#         # Step 4: Fuse with residual connection
#         fused = feature_b + a_proj

#         # Step 5: Regularize distribution (match mean/std of feature_b)
#         mean_b = feature_b.mean(dim=[1, 2], keepdim=True)
#         std_b = feature_b.std(dim=[1, 2], keepdim=True) + 1e-6

#         mean_f = fused.mean(dim=[1, 2], keepdim=True)
#         std_f = fused.std(dim=[1, 2], keepdim=True) + 1e-6

#         reg_loss = self.reg_weight * (
#             (mean_f - mean_b).pow(2).mean() + (std_f - std_b).pow(2).mean()
#         )

#         return fused, reg_loss

class CrossAttentionFusion(nn.Module):
    def __init__(self, out_channels=512, seq_len=4096):
        super().__init__()

        # Step 1: Reduce feature_a from [1, 4, 81, 1029, 2048] to [1, N, 4096]
        self.feature_a_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=1),               # reduce channels
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

# class To3D(nn.Module):
#     def __init__(self, up_h=1029, up_w=2048,in_channels=1, mid_channels=8, out_channels=4):
#         super().__init__()
#         self.up_h = up_h
#         self.up_w = up_w

#         # Channel projection per input channel (grouped conv)
#         self.channel_proj = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
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
#             # nn.BatchNorm2d(4),
#             # nn.ReLU(inplace=True)  # changed from GELU to ReLU for consistency
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
#         y = torch.sigmoid(y) 
#         y = y.view(B, d1, 4, self.up_h, self.up_w).permute(0, 2, 1, 3, 4)  # [B, 4, d1, H, W]
#         return y

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

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-14B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.fusion = CrossAttentionFusion()
        self.fusion.requires_grad_(True)
            
        self.proj = To3D()
        self.proj.requires_grad_(True)

        self.model_3D = AnySplat.from_pretrained("lhjiang/anysplat")
        self.model_3D.eval()
        self.model_3D.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        if initial_latent is not None:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - initial_latent.shape[1], *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()
            
            

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        if initial_latent is not None:
            noise_shape[1] = image_or_video_shape[1] - initial_latent.shape[1]
        else:
            noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )
