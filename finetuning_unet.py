import argparse
import itertools
import math
import os
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import gc

import PIL
from accelerate import Accelerator
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
import diffusers
print(diffusers.__version__)
from diffusers.pipelines import stable_diffusion  

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import bitsandbytes as bnb
from pathlib import Path
from torchvision import transforms
from huggingface_hub import notebook_login
from argparse import Namespace


# change accordingly for the following variables
# ==============================================
resolution=128 # change accordingly
center_crop=True, # change accordingly
learning_rate=5e-06 # change accordingly
max_train_steps=600 # change accordingly
gradient_accumulation_steps=2 # change accordingly
max_grad_norm=1.0 # change accordingly
mixed_precision="no" # set to "fp16" for mixed-precision training.
gradient_checkpointing=True # set this to True to lower the memory usage.
use_8bit_adam=True # use 8bit optimizer from bitsandbytes
seed=3434554 # change accordingly
output_dir="dreambooth-concept"
# ==============================================

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

""" ============ helper functions for preparing the data ============
"""
class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        center_crop=True
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())

        # f = open('mapping.json')
        # self.image_class_mapping = json.load(f)
        # f.close()

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        example = {}

        instance_image_path = self.instance_images_path[index]
        instance_image = Image.open(instance_image_path)

        #print(instance_image_path)

        #instance_prompt = self.image_class_mapping[instance_image_path]
        instance_prompt = "a custom icon"

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


""" ============ end of helper functions for preparing the data ============
"""
def training_function(text_encoder, vae, unet, batch_size, train_dataloader ):
    logger = get_logger(__name__)
    
    print(gradient_accumulation_steps)
    print("train set size: ", len(train_dataloader.dataset))
    print("batch size: ", batch_size)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    set_seed(seed)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # only optimize unet
        lr=learning_rate,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    #num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    num_train_epochs = 4
  
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    print("number of batches: ", math.ceil(len(train_dataloader.dataset)/batch_size))

    # Only show the progress bar once on each machine.
    #progress_bar = tqdm(range(math.ceil(len(train_dataloader.dataset)/batch_size)), disable=not accelerator.is_local_main_process)
    #progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        print("epoch: ", epoch)
        progress_bar = tqdm(range(math.ceil(len(train_dataloader.dataset)/batch_size)), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual with conditional gan
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            #if global_step >= max_train_steps:
            #    break

        accelerator.wait_for_everyone()
        print('saving model')
        torch.save(accelerator.unwrap_model(unet).state_dict(), 'unet-' + str(epoch) + '.pt')

def db_eval(batch_size, instance_image_folder):
    """
        instance_prompt (str): good description of what your object or style is, 
        together with the initializer word `sks`
        prior_preservation (bool): if you would like class of the concept (e.g.: toy, dog, painting)
        is guaranteed to be preserved. This increases the quality and helps with generalization
        at the cost of training time
        prior_preservation_class_prompt (str)
        prior_loss_weight: determins how strong the class for prior preservation

    """
    # prepare data
    # ======= instance images (desired icon style) =====
        
    # load data
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_image_folder,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop
    )

    print("dataset length: ", len(train_dataset))

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # train
    training_function(text_encoder, vae, unet, batch_size, train_dataloader)

    with torch.no_grad():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # TODO: decide whether prompts should be "a custom icon" for all 9 inputs or "a custom [apple] icon" which is subject-variant
    # change accordingly for the following variables
    # ==============================================
    instance_image_folder = "./splits/train5" # change accordingly
    # ================================================
    batch_size = 8
    db_eval(batch_size, instance_image_folder)
