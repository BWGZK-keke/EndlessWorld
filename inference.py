import argparse
import torch
import time
import os
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torch
import os
import sys
from thop import profile
from torchvision.transforms import ToPILImage, ToTensor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--baseline_checkpoint_path", type=str, default=None,
                    help="Path to the baseline (text-only) Self-Forcing DMD checkpoint. "
                         "Used to bootstrap the first chunk before 3D-aware autoregressive rollout.")
parser.add_argument("--num_extension_steps", type=int, default=30,
                    help="Number of autoregressive 3D-aware extension chunks after the initial generation.")
args = parser.parse_args()

def mse_loss(x,y):
    B = x.shape[0]
    x = x.reshape(B, -1)
    y = y.reshape(B, -1)
    loss = F.mse_loss(x, y)
    return loss
def cosine_similarity_loss(f1, f2):
    # Normalize features
    f1_normalized = F.normalize(f1, p=2, dim=1)
    f2_normalized = F.normalize(f2, p=2, dim=1)

    # Cosine similarity: (batch_size,)
    cosine_sim = torch.sum(f1_normalized * f2_normalized, dim=1)

    # Cosine similarity ranges from -1 to 1, we want to maximize it,
    # so we minimize (1 - cosine_sim)
    loss = 1 - cosine_sim.mean()

    return loss
def crop_img(img):
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

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)
    
    
# Load the model from Hugging Face
model_3D = AnySplat.from_pretrained("lhjiang/anysplat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_3D = model_3D.to(device)
model_3D.eval()
for param in model_3D.parameters():
    param.requires_grad = False

def video_to_3D_feature(model_3D,video):
    video = (video * 0.5 + 0.5).clamp(0, 1) #(1,81,3,480,832)
    video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    cropped_imgs = [crop_img(rearrange(video, 'b t h w c -> b t c h w')[0,i]) for i in range(video.shape[1])]
    cropped_imgs = torch.stack(cropped_imgs, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = cropped_imgs.shape
    # fused_feature = model_3D.inference((cropped_imgs+1)*0.5)
    feature_list = model_3D.inference((cropped_imgs+1)*0.5)
    fused_feature = torch.cat(feature_list)
    return fused_feature
    

(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    baseline_ckpt = args.baseline_checkpoint_path or "checkpoints/self_forcing_dmd.pt"
    state_dict_baseline = torch.load(baseline_ckpt, map_location="cpu")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    expected_keys = pipeline.generator.state_dict().keys()
    missing_keys, unexpected_keys = pipeline.generator.load_state_dict(state_dict_baseline['generator' if not args.use_ema else 'generator_ema'],strict=False)
    missing_keys, unexpected_keys = pipeline.fusion.load_state_dict(state_dict['fusion'],strict=False)
    # missing_keys, unexpected_keys = pipeline.proj.load_state_dict(state_dict['proj'],strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    # pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])
    # breakpoint()
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)
pipeline.fusion.to(device=gpu)
# pipeline.proj.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()
    
def downsample_video(video: torch.Tensor, max_frames: int = 81):
    """
    Evenly sample frames along dim=2 (video frame dimension) 
    so total frames <= max_frames.
    
    Args:
        video: Tensor of shape [B, C, T, H, W]
        max_frames: maximum number of frames after sampling
        
    Returns:
        Tensor of shape [B, C, T_new, H, W]
    """
    print("input_video:", video.shape)
    T = video.shape[1]
    if T <= max_frames:
        return video  # no need to downsample

    # choose evenly spaced frame indices
    indices = torch.linspace(0, T - 1, steps=max_frames).long().to(video.device)
    video_down = video.index_select(dim=1, index=indices)
    # print("########################")
    # print("down sample:",video_down.shape)
    return video_down

def normalize(x):
    B, C, H, W, D = x.shape
    x_flat = x.view(B, C, H, -1)               # [1, 4, 80, 1029*2048]
    x_norm = F.normalize(x_flat, p=2, dim=-1)  # normalize across (W,D)
    x_norm = x_norm.view(B, C, H, W, D)
    return x_norm

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output
# Optional VBench prompt remapping (extended-prompt -> short-prompt for filename).
# Only loaded when both files are available, e.g. when running the VBench evaluation.
_vbench_extended = os.path.join(os.path.dirname(__file__), "prompts/vbench/all_dimension_extended.txt")
_vbench_short = os.path.join(os.path.dirname(__file__), "prompts/vbench/all_dimension.txt")
if os.path.exists(_vbench_extended) and os.path.exists(_vbench_short):
    with open(_vbench_extended, 'r', encoding='utf-8') as f:
        old_prompts = [line.strip() for line in f if line.strip()]
    with open(_vbench_short, 'r', encoding='utf-8') as f:
        new_prompts = [line.strip() for line in f if line.strip()]
    name_dict = dict(zip(old_prompts, new_prompts))
else:
    name_dict = {}


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    idx = batch_data['idx'].item()
    model = "regular" if not args.use_ema else "ema"
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch
    prompt = batch['prompts'][0]
    # output_path = os.path.join(args.output_folder, f'{name_dict[prompt]}-{0}.mp4')
    output_path = os.path.join(args.output_folder, f'{prompt[0:100]}-{0}.mp4')
    if not os.path.exists(output_path):
        all_video = []
        num_generated_frames = 0  # Number of generated (latent) frames
        missing_keys, unexpected_keys = pipeline.generator.load_state_dict(state_dict_baseline['generator' if not args.use_ema else 'generator_ema'],strict=False)
        if args.i2v:
            # For image-to-video, batch contains image and caption
            prompt = batch['prompts'][0]  # Get caption from batch
            prompts = [prompt] * args.num_samples

            # Process the image
            image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        else:
            # For text-to-video, batch is just the text prompt
            prompt = batch['prompts'][0]
            extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
            if extended_prompt is not None:
                prompts = [extended_prompt] * args.num_samples
            else:
                prompts = [prompt] * args.num_samples
            initial_latent = None

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
        input_latents = []
        conditional_dict = pipeline.text_encoder(text_prompts=prompts)
        conditional_dict["original_embed"] = conditional_dict["prompt_embeds"].clone()
        # print(conditional_dict["prompt_embeds"].shape)
        latents = pipeline.inference(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )
        # input_latents.append(latents[:,0:18])
        each_latents = []
        output_videos = []
        # video_start = pipeline.vae.decode_to_pixel(latents, use_cache=False)
        # output_videos.append(video_start)
        ################inference projection    
        missing_keys, unexpected_keys = pipeline.generator.load_state_dict(state_dict['generator'],strict=False)
        for i in range(args.num_extension_steps):
            # # # proj_feature = normalize(proj_feature)
            # Generate 81 frames
            #############################
            # if i==0:
            video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
            start = time.time()
            static_feature_1 = video_to_3D_feature(model_3D,video[:,:,0:12]).unsqueeze(0)
            baseline_time = time.time() - start
            print(f"Baseline forward: {baseline_time:.4f} s")
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                conditional_dict["prompt_embeds"],_ = pipeline.fusion(conditional_dict["original_embed"], static_feature_1)
            #############################
            # # breakpoint()
            # if i > 0:
            #     video = pipeline.vae.decode_to_pixel(torch.cat(input_latents, dim=1), use_cache=True)
            #     conditional_dict = pipeline.text_encoder(text_prompts=prompts)
            #     conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(device)
            # print(conditional_dict["prompt_embeds"].shape)
            ############################
            # torch.cuda.empty_cache()
            
            # if i ==0:
            # print("###################################")
            # # video_cat = torch.cat(output_videos,dim=1)
            # if i > 0:
            #     print(torch.cat(input_latents,dim=1).shape)
            #     video_cat = pipeline.vae.decode_to_pixel(torch.cat(input_latents,dim=1), use_cache=True)
            #     # input_video = downsample(video_cat)
            # else:
            #     video_cat = pipeline.vae.decode_to_pixel(latents, use_cache=False)
            #     input_video = video_cat
            # # # # input_video = video_cat[:,-81:]
            # input_video = torch.cat([video_cat[:,0:1],downsample_video(video_cat,3*4),video_cat[:,-17*4:]],dim=1)
            # # input_video = downsample(video_cat)
            # static_feature_1 = video_to_3D_feature(model_3D,input_video).unsqueeze(0)
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # #     conditional_dict["prompt_embeds"],_ = pipeline.fusion(conditional_dict["original_embed"], static_feature_1[:,:,1:-4*3])
            #     _, proj_bias = pipeline.proj(static_feature_1)
            
            ######################################
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            #     fused_feature = pipeline.fusion(conditional_dict["prompt_embeds"], fused_feature[:,:,:])
            # conditional_dict["prompt_embeds"] = fused_feature
            # torch.cuda.empty_cache()
            #####################
            # video = (video * 0.5 + 0.5).clamp(0, 1) 
            # video = (video -0.5)/0.5
            # rearrange_video = rearrange(video, 'b t c h w -> b c t h w').to(device=device, dtype=torch.bfloat16)
            # #[batch_size, num_channels, num_frames, height, width]
            # initial_latent_1 = pipeline.vae.encode_to_latent(rearrange_video[:,:,-69:]).to(device=device, dtype=torch.bfloat16)
            # initial_latent_last = pipeline.vae.encode_to_latent((rearrange_video[:,:,-8:]-0.5)/0.5).to(device=device, dtype=torch.bfloat16)
            #initial_latent = initial_latent_1#torch.cat([initial_latent_1, initial_latent_last], dim=1)
            # print(initial_latent.shape)
            #############################################
            ##############################################
            sampled_noise = torch.randn(
                [args.num_samples, 3, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
            if i == 0:
                initial_latent=torch.cat([latents[:,0:18]],dim=1)
            else:
                initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-17:]],dim=1)
            latents = pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=True,
                # initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-17:]+proj_bias[:,-17:]],dim=1),
                # initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-17:]],dim=1),
                # initial_latent=torch.cat([torch.cat(input_latents,dim=1)[:,-18:]],dim=1),
                initial_latent=initial_latent,
                # initial_latent=torch.cat([input_latents[0][:,0:3],input_latents[-1][:,-15:]],dim=1),
                low_memory=low_memory,
            )
            print(latents.shape)
            # each_latents.append(latents)
            # torch.cuda.empty_cache()
            if i==0:
                input_latents.append(latents)
            else:
                input_latents.append(latents[:,-3:])
            ################################################
            ###############################################
            # video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
            # breakpoint()
            # static_feature = video_to_3D_feature(model_3D,video).unsqueeze(0)[:,:,1:]
            # fused_feature2 = pipeline.fusion(latents[:,1:])
            # print("ori fused_feature:", fused_feature2.mean(), fused_feature2.std(), fused_feature2.shape)
            # print("ori 3D_feature:", static_feature.mean(), static_feature.std(),static_feature.shape)
            # static_feature = normalize(static_feature)
            # fused_feature2 = normalize(fused_feature2)
            # mse_loss_1 = mse_loss(static_feature, fused_feature2)
            # similar_loss = cosine_similarity_loss(static_feature, fused_feature2)
            # print("fused_feature:", fused_feature2.mean(), fused_feature2.std(), fused_feature2.shape)
            # print("3D_feature:", static_feature.mean(), static_feature.std(),static_feature.shape)
            # print("mse:", mse_loss_1)
            # print("similar:", similar_loss)
            # conditional_dict = pipeline.text_encoder(text_prompts=prompts)
            # conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(device)

            # video = pipeline.vae.decode_to_pixel(torch.cat(input_latents,dim=1), use_cache=True)
            # ################
            # fused_feature = video_to_3D_feature(model_3D,downsample_video(video)).unsqueeze(0)
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            #     fused_feature = pipeline.fusion(conditional_dict["prompt_embeds"], fused_feature[:,:,:])
            # conditional_dict["prompt_embeds"] = fused_feature
            ############################
            
            # video = (video * 0.5 + 0.5).clamp(0, 1) 
            # video = (video -0.5)/0.5
            # rearrange_video = rearrange(video, 'b t c h w -> b c t h w').to(device=device, dtype=torch.bfloat16)
            # #[batch_size, num_channels, num_frames, height, width]
            # initial_latent_1 = pipeline.vae.encode_to_latent(rearrange_video[:,:,-9:]).to(device=device, dtype=torch.bfloat16)
            # # initial_latent_last = pipeline.vae.encode_to_latent((rearrange_video[:,:,-8:]-0.5)/0.5).to(device=device, dtype=torch.bfloat16)
            # initial_latent = initial_latent_1#torch.cat([initial_latent_1, initial_latent_last], dim=1)

            # proj_feature = pipeline.proj(torch.cat(input_latents,dim=1))
            # proj_feature = normalize(proj_feature)
            # conditional_dict["prompt_embeds"],_ = pipeline.fusion(conditional_dict["prompt_embeds"], proj_feature)

            # proj_feature = pipeline.proj(input_latents[-1][:,-8:])
            # conditional_dict["prompt_embeds"],_ = pipeline.fusion(conditional_dict["original_embed"], proj_feature)
            # video_now = pipeline.vae.decode_to_pixel(torch.cat(input_latents, dim=1)[:,-21:], use_cache=False)
            # video_cat = torch.cat([video_start[:,0:1],video_now[:,-20*4:]],dim=1)
            # video_cat = pipeline.vae.decode_to_pixel(torch.cat(input_latents,dim=1), use_cache=True)
            # input_video = torch.cat([video_cat[:,0:1],downsample_video(video_cat,18*4),video_cat[:,-2*4:]],dim=1)
            # static_feature_1 = video_to_3D_feature(model_3D,input_video).unsqueeze(0)
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # #     conditional_dict["prompt_embeds"],_ = pipeline.fusion(conditional_dict["original_embed"], static_feature_1)
            #     _, proj_bias = pipeline.proj(static_feature_1)
            
            sampled_noise = torch.randn(
                [args.num_samples, 18, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
            latents = pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=True,
                # initial_latent=torch.cat([input_latents[0][:,0:1],input_latents[-1][:,-2:]+proj_bias[:,-2:]],dim=1),
                initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-2:]],dim=1),
                # initial_latent=torch.cat([torch.cat(input_latents,dim=1)[:,0:3]],dim=1),
                # initial_latent=initial_latent,
                low_memory=low_memory,
            )
            # each_latents.append(latents)
            input_latents.append(latents[:,-18:])
            # video_now = pipeline.vae.decode_to_pixel(torch.cat(input_latents, dim=1)[:,-21:], use_cache=False)
            # output_videos.append(video_now)
            torch.cuda.empty_cache()
        
        
        # for i in range(5):
        #     video = pipeline.vae.decode_to_pixel(latents, use_cache=True)
        #     # breakpoint()
        #     # if i ==0:
        #     static_feature = video_to_3D_feature(model_3D,video).unsqueeze(0)
        #     print("static_feature",static_feature.mean(), static_feature.std())
        #     # else:
        #     #     video = pipeline.vae.decode_to_pixel(torch.cat(input_latents,dim=1), use_cache=True)
        #     #     static_feature_tmp = video_to_3D_feature(model_3D,video).unsqueeze(0)
        #     #     print("video:",video.shape,"static_feature_tmp:", static_feature_tmp.shape)
        #     conditional_dict = pipeline.text_encoder(text_prompts=prompts)
        #     conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(device)

        #     video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
        #     # video = (video * 0.5 + 0.5).clamp(0, 1) 
        #     # video = (video -0.5)/0.5
        #     rearrange_video = rearrange(video, 'b t c h w -> b c t h w').to(device=device, dtype=torch.bfloat16)
        #     #[batch_size, num_channels, num_frames, height, width]
        #     initial_latent_1 = pipeline.vae.encode_to_latent(rearrange_video[:,:,-9:]).to(device=device, dtype=torch.bfloat16)
        #     # initial_latent_last = pipeline.vae.encode_to_latent((rearrange_video[:,:,-8:]-0.5)/0.5).to(device=device, dtype=torch.bfloat16)
        #     initial_latent = initial_latent_1#torch.cat([initial_latent_1, initial_latent_last], dim=1)
        #     print("video:", initial_latent.mean(), initial_latent.shape)
        #     with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        #         fused_feature = pipeline.fusion(static_feature[:,:,0:9],initial_latent)
            # sampled_noise = torch.randn(
            #     [args.num_samples, 18, 16, 60, 104], device=device, dtype=torch.bfloat16
            # )
            # latents = pipeline.inference(
            #     noise=sampled_noise,
            #     conditional_dict=conditional_dict,
            #     return_latents=True,
            #     initial_latent=fused_feature,
            #     low_memory=low_memory,
            # )
            # if i ==0:
            #     input_latents.append(latents)
            # else:
            #     input_latents.append(latents[:,-18:])
            
        latents = torch.cat(input_latents, dim=1)
        # print(latents.shape)
        # conditional_dict = pipeline.text_encoder(text_prompts=prompts)
        video = pipeline.vae.decode_to_pixel(latents, use_cache=True)
        # video = pipeline.vae.decode_to_pixel(input_latents[0], use_cache=False)
        # input_video = downsample_video(video,latents.shape[1])
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     static_feature_1 = video_to_3D_feature(model_3D,input_video).unsqueeze(0)
        #     _, proj_bias = pipeline.proj(static_feature_1)
        # print("proj_bias shape:", proj_bias.shape)

        # input_latents[0][:,0:1] = input_latents[0][:,0:1]
        # input_latents_v2 = [input_latents[0][:,0:3]]
        # # input_latents_v2 = [input_latents[0][:,0:1],input_latents[0][:,1:3]+proj_bias[:,1:3]]
        # sampled_noise = torch.randn(
        #             [args.num_samples, 18, 16, 60, 104], device=device, dtype=torch.bfloat16
        #         )
        # dim_bias = torch.cat(input_latents_v2,dim=1).shape[1]
        # latents = pipeline.inference(
        #     noise=sampled_noise,
        #     conditional_dict=conditional_dict,
        #     return_latents=True,
        #     initial_latent=torch.cat([input_latents_v2[0][:,0:1],torch.cat(input_latents_v2,dim=1)[:,-2:]],dim=1),
        #     # initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-17:]],dim=1),
        #     low_memory=low_memory,
        # )
        # print(latents.shape)
        # input_latents_v2.append(latents[:,-18:])
        # for i in range(11):
        #     sampled_noise = torch.randn(
        #         [args.num_samples, 3, 16, 60, 104], device=device, dtype=torch.bfloat16
        #     )
        #     dim_bias = torch.cat(input_latents_v2,dim=1).shape[1]
        #     latents = pipeline.inference(
        #         noise=sampled_noise,
        #         conditional_dict=conditional_dict,
        #         return_latents=True,
        #         initial_latent=torch.cat([input_latents_v2[0][:,0:1],torch.cat(input_latents_v2,dim=1)[:,-17:]],dim=1),
        #         # initial_latent=torch.cat([input_latents[0][:,0:1],torch.cat(input_latents,dim=1)[:,-17:]],dim=1),
        #         low_memory=low_memory,
        #     )
        #     print(latents.shape)
        #     input_latents_v2.append(latents[:,-3:])
        #     dim_bias = torch.cat(input_latents_v2,dim=1).shape[1]
        #     sampled_noise = torch.randn(
        #         [args.num_samples, 18, 16, 60, 104], device=device, dtype=torch.bfloat16
        #     )
        #     print("dim_bias:",dim_bias)
        #     latents = pipeline.inference(
        #         noise=sampled_noise,
        #         conditional_dict=conditional_dict,
        #         return_latents=True,
        #         initial_latent=torch.cat([input_latents_v2[0][:,0:1],torch.cat(input_latents_v2,dim=1)[:,-2:]],dim=1),
        #         low_memory=low_memory,
        #     )
        #     input_latents_v2.append(latents[:,-18:])
        #     torch.cuda.empty_cache()
        
        # latents = torch.cat(input_latents_v2, dim=1)
        # video = pipeline.vae.decode_to_pixel(latents, use_cache=True)
        video = (video * 0.5 + 0.5).clamp(0, 1) #(1,81,3,480,832)
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            model = "regular" if not args.use_ema else "ema"
            for seed_idx in range(args.num_samples):
                # All processes save their videos
                # if args.save_with_index:
                #     output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
                # else:
                #     output_path = os.path.join(args.output_folder, f'{name_dict[prompt]}-{seed_idx}.mp4')
                write_video(output_path, video[seed_idx], fps=16)
                