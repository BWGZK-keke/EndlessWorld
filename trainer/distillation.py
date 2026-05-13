import gc
import logging
from torch.cuda.amp import autocast
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
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os

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
import torch.nn.functional as F
def normalize(x):
    B, C, H, W, D = x.shape
    x_flat = x.view(B, C, H, -1)               # [1, 4, 80, 1029*2048]
    x_norm = F.normalize(x_flat, p=2, dim=-1)  # normalize across (W,D)
    x_norm = x_norm.view(B, C, H, W, D)
    return x_norm

def cosine_similarity_loss(f1, f2):
    # Normalize features
    # print("cosine_sim:", f1.shape, f2.shape)
    loss = torch.nn.functional.cosine_similarity(f1, f2, dim=1, eps=1e-8)
    # Cosine similarity ranges from -1 to 1, we want to maximize it,
    # so we minimize (1 - cosine_sim)
    loss = - loss.mean()

    return loss

def compute_mmd(x, y, sigma=1.0):
    """
    x, y: [B, N, D] — batch of N samples, D-dimensional
    """
    def gaussian_kernel(a, b, sigma):
        a = a.unsqueeze(2)  # [B, N, 1, D]
        b = b.unsqueeze(1)  # [B, 1, M, D]
        return torch.exp(-((a - b) ** 2).mean(-1) / (2 * sigma ** 2))  # [B, N, M]
    def flatten_feature(tensor):
        # tensor: [1, T, D, H, C] → [1, D, T*H*C]
        B, T, D, H, C = tensor.shape
        return tensor.permute(0, 2, 1, 3, 4).reshape(B, D, -1)  # [B, D, T*H*C]
    x = flatten_feature(x)  # [1, d1, D]
    y = flatten_feature(y)  # [1, d3, D]
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

def mse_loss(x,y):
    B = x.shape[0]
    x = x.reshape(B, -1)
    y = y.reshape(B, -1)
    loss = F.mse_loss(x, y)
    return loss

    
class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
           cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
           cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )
        
        self.model.model_3D = fsdp_wrap(
            self.model.model_3D,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )

        self.model.proj = fsdp_wrap(
            self.model.proj,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
           cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )
        

        self.model.fusion = fsdp_wrap(
            self.model.fusion,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
           cpu_offload=getattr(config, "text_encoder_cpu_offload", True)
        )
        
        # self.model_3D.eval()
    
        # for param in self.model_3D.parameters():
        #     param.requires_grad = False

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        self.fusion_optimizer = torch.optim.AdamW(
            [param for param in self.model.fusion.parameters()
             if param.requires_grad],
            lr=config.lr_fusion,
            betas=(config.beta1_fusion, config.beta2_fusion),
            weight_decay=config.weight_decay
        )
        
        self.proj_optimizer = torch.optim.AdamW(
            [param for param in self.model.proj.parameters()
             if param.requires_grad],
            lr=config.lr_proj,
            betas=(config.beta1_proj, config.beta2_proj),
            weight_decay=config.weight_decay
        )


        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
        ##############################################################################################################
        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "model" in state_dict:
                if "generator" in state_dict:
                    state_dict = state_dict["generator"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                
                self.model.generator.load_state_dict(
                    state_dict, strict=True
                )
            elif "generator_ema" in state_dict:
                # This method will copy EMA weights into the generator
                # missing_keys, unexpected_keys = self.generator_ema.load_state_dict(state_dict)

                # if missing_keys or unexpected_keys:
                #     print(f"Missing keys: {missing_keys}")
                #     print(f"Unexpected keys: {unexpected_keys}")
                
                # self.generator_ema.load_state_dict(state_dict["generator_ema"])
                self.generator_ema.load_state_dict(state_dict["generator"])
                self.model.generator.load_state_dict(self.generator_ema.state_dict())

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

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
    
    # def video_to_3D_feature(self, video):
    #     video = (video * 0.5 + 0.5).clamp(0, 1)*255 #(1,81,3,480,832)
    #     video = rearrange(video, 'b t c h w -> b t h w c')
    #     cropped_imgs = [self.crop_img(rearrange(video, 'b t h w c -> b t c h w')[0,i]) for i in range(video.shape[1])]
    #     cropped_imgs = torch.stack(cropped_imgs, dim=0).unsqueeze(0) # [1, K, 3, 448, 448]
    #     b, v, _, h, w = cropped_imgs.shape
    #     feature_list = self.model_3D.inference((cropped_imgs+1)*0.5)
    #     fused_feature = torch.cat(feature_list)
    #     fused_feature  = (fused_feature  -  fused_feature.min()) / ( fused_feature.max() -  fused_feature.min() + 1e-8)
    #     return fused_feature

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        fusion_state_dict = fsdp_state_dict(
            self.model.fusion)
        proj_state_dict = fsdp_state_dict(
            self.model.proj)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "fusion":  fusion_state_dict,
                "proj":  proj_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "fusion":  fusion_state_dict,
                "proj":  proj_state_dict,
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)
        if train_generator:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.fusion_optimizer.zero_grad(set_to_none=True)
            self.proj_optimizer.zero_grad(set_to_none=True)
        else:
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.proj_optimizer.zero_grad(set_to_none=True)
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        # self.generate_video(text_prompts)
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        conditional_dict["original_embeds"] = conditional_dict["prompt_embeds"].clone()
        unconditional_dict["original_embeds"] = unconditional_dict["prompt_embeds"].clone()
        rand_tensor = torch.tensor([random.randint(0, 6)], device="cuda")
        rand_tensor_latent = torch.tensor([random.randint(1,20)], device="cuda")
        if (rand_tensor - 1) > 1:
            start_index = torch.randint(0, (rand_tensor - 1).item(), (1,), device="cuda")
        else:
            start_index = torch.tensor([0], device="cuda")
        if dist.is_initialized():
            dist.broadcast(rand_tensor, src=0)
            dist.broadcast(rand_tensor_latent, src=0)
            dist.broadcast(start_index, src=0)

        with torch.no_grad():
            _, _, video_latents1 = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            video = self.model.vae.decode_to_pixel(video_latents1.detach(), use_cache=False)
            # static_feature_1 = self.video_to_3D_feature(video).unsqueeze(0) #[1, 81, 1029, 2048]
            # rearrange_video = rearrange(video, 'b t c h w -> b c t h w')
            # if rand_tensor > 0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # conditional_dict["insert_latent_input"] = self.model.vae.encode_to_latent(rearrange_video[:,:,-((rand_tensor-start_index)*3-1)*4-1:].cuda())
                # static_feature_1 = self.video_to_3D_feature(video[:,-((rand_tensor-start_index)*3-1)*4-1:]).unsqueeze(0) #[1, 81, 1029, 2048]
                #static_feature_1 = self.video_to_3D_feature(video).unsqueeze(0) #[1, 81, 1029, 2048]
                static_feature_1 = self.video_to_3D_feature(video[:, 0:(rand_tensor*3-1)*4+1]).unsqueeze(0) #[1, 81, 1029, 2048]
                static_feature_0 = self.video_to_3D_feature(video).unsqueeze(0)
            
                    
        # print(video_latents1[:,0:3].shape, static_feature_1[:,:,0:9].shape)
        
        # fused_feature = self.model.fusion(static_feature_1[:,:,0:(rand_tensor*3-1)*4+1],video_latents1[:,0:rand_tensor*3])
        #############################################################
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     proj_feature = self.model.proj(video_latents1[:,1:rand_tensor_latent+1].detach())
        #     cosine_loss_1 =cosine_similarity_loss(static_feature_1[:,:,1:rand_tensor_latent*4+1].detach(), proj_feature)
        #     mse_loss_1 = F.mse_loss(static_feature_1[:,:,1:rand_tensor_latent*4+1].detach(), proj_feature)
        #     similarity_loss1 = cosine_loss_1+mse_loss_1
        ################################################################
        
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #     if rand_tensor > 0:
        #         _, proj_bias = self.model.proj(static_feature_1)
        #     cosine_loss_1 =0#cosine_similarity_loss(proj_feature, video_latents1)
        #     mse_loss_1 = 0#F.mse_loss(proj_feature, video_latents1)
            # similarity_loss1 = cosine_loss_1+mse_loss_1
    
        # similarity_loss1.backward()
        # self.proj_optimizer.step()
        # self.proj_optimizer.zero_grad(set_to_none=True)

        # del fused_feature
        
        ##########################################
        # if rand_tensor > 1:
        #     with torch.no_grad():
        #         if self.step < 1000:
        #             fused_embed,_ = self.model.fusion(conditional_dict["prompt_embeds"], static_feature_1[:,:,1:(rand_tensor*3-1)*4+1].detach())
        #         else:
        #             proj_feature = self.model.proj(video_latents1[:,1:rand_tensor*3].detach())
        #             fused_embed,_ = self.model.fusion(conditional_dict["prompt_embeds"], proj_feature.detach())
        # else:
        #     fused_embed = conditional_dict["prompt_embeds"]
        ############################################
        ##########################################
        if rand_tensor > 1:
            fused_embed,_ = self.model.fusion(conditional_dict["prompt_embeds"], static_feature_1.detach())
        else:
           fused_embed = conditional_dict["prompt_embeds"]
        # fused_embed = conditional_dict["prompt_embeds"]
        ############################################
        
        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # if random.random() < 0.5:
            #     generator_loss, generator_log_dict, video_latents = self.model.generator_loss(
            #         image_or_video_shape=image_or_video_shape,
            #         conditional_dict=conditional_dict,
            #         unconditional_dict=unconditional_dict,
            #         clean_latent=clean_latent,
            #         initial_latent=image_latent if self.config.i2v else None
            #     )

            #     generator_loss.backward()
            #     generator_grad_norm = self.model.generator.clip_grad_norm_(
            #         self.max_grad_norm_generator)

            #     generator_log_dict.update({"generator_loss": generator_loss,
            #                             "generator_grad_norm": generator_grad_norm})

            #     return generator_log_dict
            # else:

            # if rand_tensor.item() < 0.5:
            #     with torch.no_grad():
            #         _, _, video_latents = self.model.generator_loss(
            #             image_or_video_shape=image_or_video_shape,
            #             conditional_dict=conditional_dict,
            #             unconditional_dict=unconditional_dict,
            #             clean_latent=clean_latent,
            #             initial_latent=image_latent if self.config.i2v else None
            #         )

            #         video = self.model.vae.decode_to_pixel(video_latents.detach(), use_cache=False)
            #         static_feature_1 = self.video_to_3D_feature(video).unsqueeze(0) #[4, 81, 1029, 2048]
            #         fused_feature = self.model.fusion(conditional_dict["prompt_embeds"].clone(), static_feature_1)
            #         temp_prompt_embeds = fused_feature.detach()
            #         del video
            #         del fused_feature
            #         del video_latents
            #         torch.cuda.empty_cache()
            #     conditional_dict["prompt_embeds"] = temp_prompt_embeds
            # else:
            #     conditional_dict["prompt_embeds"] = original_prompt_embeds 

            # generator_loss, generator_log_dict, video_latents = self.model.generator_loss(
            #     image_or_video_shape=image_or_video_shape,
            #     conditional_dict=conditional_dict,
            #     unconditional_dict=unconditional_dict,
            #     clean_latent=clean_latent,
            #     initial_latent=image_latent if self.config.i2v else None
            # )
            
            ## test_generation
            # if rand_tensor > 0:
            #     # conditional_dict["insert_latent"] = video_latents1[:,start_index*3:rand_tensor*3]
            #     # conditional_dict["insert_latent"] = video_latents1[:,0:1]
            #     generator_loss, generator_log_dict, video_latents2 = self.model.generator_loss(
            #         image_or_video_shape=image_or_video_shape,
            #         conditional_dict=conditional_dict,
            #         unconditional_dict=unconditional_dict,
            #         clean_latent=clean_latent,
            #         initial_latent=torch.cat([video_latents1[:,0:1],video_latents1[:,1+start_index*3:rand_tensor*3]],dim=1)#for tmp
            #         # initial_latent=video_latents1[:,0:rand_tensor*3] # torch.Size([1, 3, 16, 60, 104] # for after iccv without tmp
            #         # initial_latent=image_latent if self.config.i2v else Noneg
            #     )
            # else:
            #     generator_loss, generator_log_dict, video_latents2 = self.model.generator_loss(
            #         image_or_video_shape=image_or_video_shape,
            #         conditional_dict=conditional_dict,
            #         unconditional_dict=unconditional_dict,
            #         clean_latent=clean_latent,
            #         initial_latent=None # torch.Size([1, 3, 16, 60, 104]
            #         # initial_latent=image_latent if self.config.i2v else Noneg
            #     )
            # # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # #     proj_feature2 = self.model.proj(video_latents2[:,rand_tensor*3:])
            # #     proj_feature1 = self.model.proj(video_latents1[:,rand_tensor*3:].detach())
            # #     # mse_loss_2 = mse_loss(static_feature_1[:,:,1:], proj_feature2)
            # #     cosine_loss_2 = cosine_similarity_loss(proj_feature1, proj_feature2)
            # #     similarity_loss2 = cosine_loss_2
                    
            # # print("similarity loss2:", cosine_loss_2.item())
            # # print("reg loss:", reg_loss.item())
            # torch.cuda.empty_cache()


            # (generator_loss).backward() #similarity_loss2+reg_loss
            # generator_grad_norm = self.model.generator.clip_grad_norm_(
            #     self.max_grad_norm_generator)
            # self.generator_optimizer.step()

            # self.generator_optimizer.zero_grad(set_to_none=True)
            # self.fusion_optimizer.zero_grad(set_to_none=True)
            
            torch.cuda.empty_cache()
            
            conditional_dict["prompt_embeds"] = fused_embed
            if rand_tensor > 0:
                # dim1 = conditional_dict["insert_latent_input"].shape[1]
                # conditional_dict["insert_latent"] = video_latents1[:,0:1]
                # conditional_dict["insert_latent"] = video_latents1[:,0:rand_tensor*3]
                generator_loss2, generator_log_dict, video_latents2 = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    # initial_latent=torch.cat([video_latents1[:,0:1]+proj_bias[:,0:1],video_latents1[:,1:rand_tensor*3]+proj_bias[:,1:rand_tensor*3]],dim=1),
                    # initial_latent=torch.cat([video_latents1[:,0:1],video_latents1[:,1+start_index*3:rand_tensor*3]],dim=1)
                    #initial_latent=torch.cat([video_latents1[:,0:1]+proj_bias[:,0:1],video_latents1[:,1+start_index*3:rand_tensor*3]+proj_bias[:,1+start_index*3:rand_tensor*3]],dim=1)
                    initial_latent=video_latents1[:,0:rand_tensor*3] # torch.Size([1, 3, 16, 60, 104]
                    # initial_latent=torch.cat([conditional_dict["insert_latent_input"][:,0:1],conditional_dict["insert_latent_input"][:,-(dim1-1):]+proj_bias[:,-(dim1-1):]],dim=1),
                    # initial_latent=image_latent if self.config.i2v else Noneg
                )
                # start_position = torch.cat([video_latents1[:,0:1],video_latents1[:,1+start_index*3:rand_tensor*3]],dim=1).shape[1]
                start_position = torch.cat([video_latents1[:,0:1],video_latents1[:,1:rand_tensor*3]],dim=1).shape[1]
            else:
                generator_loss2, generator_log_dict, video_latents2 = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=None # torch.Size([1, 3, 16, 60, 104]
                    # initial_latent=image_latent if self.config.i2v else Noneg
                )
                start_position = 0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if start_position > 0:
                    video_latent2_fuse = torch.cat([video_latents1[:,0:rand_tensor*3],video_latents2[:,start_position:start_position+(21-rand_tensor*3)]],dim=1)
                else:
                    video_latent2_fuse = video_latents2
                
                video2 = self.model.vae.decode_to_pixel(video_latent2_fuse.detach(), use_cache=False)
                static_feature_2 = self.video_to_3D_feature(video2).unsqueeze(0) #[4, 81, 1029, 2048]
                # cosine_loss_2 =cosine_similarity_loss(static_feature_1, static_feature_2)
                
                mse_loss_2 = F.mse_loss(static_feature_0,static_feature_2)
                similarity_loss2 = mse_loss_2 
                
            # (generator_loss2+similarity_loss2+similarity_loss1).backward()
            (generator_loss2+similarity_loss2).backward()
            proj_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.proj.parameters(), 100
                )
            # print("proj_grad_norm:", proj_grad_norm)
        
            fusion_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.fusion.parameters(), 100
            )
            # print("fusion_grad_norm:", fusion_grad_norm)
            
            self.fusion_optimizer.step()
            self.generator_optimizer.step()
                
            self.proj_optimizer.step()

            generator_log_dict.update({#"generator_loss": generator_loss,
                                       "generator_loss2": generator_loss2,
                                        "cosine_loss1": 0,
                                        "mse_loss1": 0,
                                        "similarity_loss2": similarity_loss2,
                                    #"generator_grad_norm": generator_grad_norm,
                                    "proj_grad_norm": proj_grad_norm,
                                    "fusion_grad_norm": fusion_grad_norm})

            return generator_log_dict
        
        else:
            generator_log_dict = {} 

        # if rand_tensor > 0:
        #     critic_loss, critic_log_dict = self.model.critic_loss(
        #         image_or_video_shape=image_or_video_shape,
        #         conditional_dict=conditional_dict,
        #         unconditional_dict=unconditional_dict,
        #         clean_latent=clean_latent,
        #         initial_latent=torch.cat([video_latents1[:,0:rand_tensor*3]],dim=1)
        #     )
        # else:
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=None
        )
            
        conditional_dict["prompt_embeds"] = fused_embed.detach()

        # if rand_tensor > 0:
        #     critic_loss2, critic_log_dict = self.model.critic_loss(
        #         image_or_video_shape=image_or_video_shape,
        #         conditional_dict=conditional_dict,
        #         unconditional_dict=unconditional_dict,
        #         clean_latent=clean_latent,
        #         # initial_latent=torch.cat([video_latents1[:,0:rand_tensor*3]],dim=1)
        #         initial_latent=torch.cat([video_latents1[:,0:1]+proj_bias[:,0:1],video_latents1[:,1+start_index*3:rand_tensor*3]+proj_bias[:,1+start_index*3:rand_tensor*3]],dim=1)
        #     )
        # else:
        #     critic_loss2, critic_log_dict = self.model.critic_loss(
        #         image_or_video_shape=image_or_video_shape,
        #         conditional_dict=conditional_dict,
        #         unconditional_dict=unconditional_dict,
        #         clean_latent=clean_latent,
        #         initial_latent=None
        #     )

        (critic_loss).backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                # "critic_loss2": critic_loss2,
                                "critic_grad_norm": critic_grad_norm})
        self.critic_optimizer.step()

        return critic_log_dict

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
    

    def downsample_video_4to1(self,video):
        # print(video.shape)
        T = video.shape[1]
        n = (T - 1) // 4  # compute n
        sampled = [0] + [1 + 4 * i for i in range(n)]
        indices = torch.tensor(sampled, device=video.device).long()
        video_down = video.index_select(dim=1, index=indices)
        # print(video_down.shape)
        return video_down

    def video_to_3D_feature(self, video):
        video = self.downsample_video_4to1(video)
        video = (video * 0.5 + 0.5).clamp(0, 1) #(1,81,3,480,832)
        video = rearrange(video, 'b t c h w -> b t h w c')
        cropped_imgs = [self.crop_img(rearrange(video, 'b t h w c -> b t c h w')[0,i]) for i in range(video.shape[1])]
        cropped_imgs = torch.stack(cropped_imgs, dim=0).unsqueeze(0) # [1, K, 3, 448, 448]
        fused_feature = self.model.model_3D.inference((cropped_imgs+1)*0.5)
        # fused_feature = torch.cat(feature_list)
        # fused_feature  = 0.2+0.6*(fused_feature  -  fused_feature.min()) / ( fused_feature.max() -  fused_feature.min() + 1e-8) #add weight for fun
        return fused_feature

    def generate_video(self, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = self.model.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = self.model.inference_pipeline(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                # self.generator_optimizer.zero_grad(set_to_none=True)
                # self.fusion_optimizer.zero_grad(set_to_none=True)
                # self.proj_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                # print("generator_loss", generator_log_dict["generator_loss"].mean().item())
                print("generator_loss2", generator_log_dict["generator_loss2"].mean().item())
                # print("cosine_loss1",  generator_log_dict["cosine_loss1"].mean().item())
                # print("mse_loss1", generator_log_dict["mse_loss1"].mean().item())
                print("similarity_loss2", generator_log_dict["similarity_loss2"].mean().item())
                # print("generator_grad_norm", generator_log_dict["generator_grad_norm"].mean().item())
                print("proj_grad_norm", generator_log_dict["proj_grad_norm"].mean().item())
                print("fusion_grad_norm", generator_log_dict["fusion_grad_norm"].mean().item())
            
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            # self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            print("critic_loss", critic_log_dict["critic_loss"].mean().item())
            # print("critic_loss2", critic_log_dict["critic_loss2"].mean().item())
            print("critic_grad_norm", critic_log_dict["critic_grad_norm"].mean().item())
            # self.critic_optimizer.step()
            # self.proj_optimizer.step()
            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            # "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            # "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
