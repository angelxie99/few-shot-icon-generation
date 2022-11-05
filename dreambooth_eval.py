import argparse
import itertools
import math
import os
#from contextlib import nullcontext
import random

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
#from stable_diffusion import StableDiffusionSafetyChecker

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
        instance_prompts,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=True,
        num_images = 9
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())[:num_images]
        self.num_instance_images = num_images
        self.instance_prompts = instance_prompts
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())[:num_images]
            self.num_class_images = num_images
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_prompt = self.instance_prompts[index % self.num_instance_images]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
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

def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")

""" ============ end of helper functions for preparing the data ============
"""
def training_function(text_encoder, vae, unet, batch_size, with_prior, prior_weight,  train_dataloader ):
    logger = get_logger(__name__)
    print(gradient_accumulation_steps)
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
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
  
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
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

                if with_prior:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)
                    # Compute instance loss
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()
                    # Add the prior loss to the instance loss.
                    loss = loss + prior_weight * prior_loss
                else:
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

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
    
    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
            ),
            safety_checker=stable_diffusion.StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(output_dir)


def db_eval(instance_prompts, prior_preservation, prior_preservation_class_prompt, 
            num_images, batch_size, prior_loss_weight, prior_preservation_class_folder,
            instance_image_folder, num_samples, output_prompts, output_image_dir):
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
    
    # ======= class images (desired class) =====
    if prior_preservation:
        class_images_dir = Path(prior_preservation_class_folder)
        if not os.path.exists(prior_preservation_class_folder):
            os.mkdir(prior_preservation_class_folder)
        current_num_images = len(list(class_images_dir.iterdir()))
        if current_num_images < num_images:
            pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path, revision="fp16", torch_dtype=torch.float16).to("cuda")
            pipeline.enable_attention_slicing()
            pipeline.set_progress_bar_config(disable=True)
            sample_dataset = PromptDataset(prior_preservation_class_prompt, num_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size)
            for example in tqdm(sample_dataloader, desc="Generating class images"):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    image.save(f"{prior_preservation_class_folder}/{example['index'][i] + current_num_images}.jpg")
            pipeline = None
            gc.collect()
            del pipeline
        with torch.no_grad():
            torch.cuda.empty_cache()
        
    # load data
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_image_folder,
        instance_prompts=instance_prompts,
        class_data_root=prior_preservation_class_folder,
        class_prompt=prior_preservation_class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
        num_images = 9
    )


    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # concat class and instance examples for prior preservation
        if prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

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
    accelerate.notebook_launcher(training_function,num_processes = 1,  args=(text_encoder, vae, unet, batch_size, prior_preservation, prior_loss_weight,  train_dataloader))
    with torch.no_grad():
        torch.cuda.empty_cache()

    # set up pipeline
    try:
        pipe
    except NameError:
        pipe = StableDiffusionPipeline.from_pretrained(
            output_dir,
            torch_dtype=torch.float16,
        ).to("cuda")
    
    # inference
    for prompt in output_prompts:
        images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=100, guidance_scale=7.5).images
        prompt_str = "_".join(prompt.split())
        [image.save(f"{output_image_dir}/{prompt_str}{i}.jpeg") for i, image in enumerate(images)]

def generate_output_prompts(as_type, prompts):
    if as_type:
        return ["a "+str(prompt)+" as a custom icon" for prompt in prompts]
    else:
        return ["a custom icon "+str(prompt) for prompt in prompts]


if __name__ == '__main__':
    # TODO: decide whether prompts should be "a custom icon" for all 9 inputs or "a custom [apple] icon" which is subject-variant
    prior_preservation_class_prompt = "colored icon style" 
    num_images = 9
    # change accordingly for the following variables
    # ==============================================
    instance_prompts = ["a custom icon"]*9  # change accordingly
    num_samples = 1 # change accordingly
    object_list = ["pizza", "peach", "phone", "laptop", "umbrella", "cupcake", "pancake"] # change accordingly
    output_prompts = generate_output_prompts(True, object_list) # change (if True, "a star as a custom icon")
    output_image_dir = "./output_images/output_images_600" # change accordingly
    prior_preservation_class_folder = None # change accordingly
    prior_preservation = prior_preservation_class_folder is not None # change accordingly (False at current stage)
    instance_image_folder = "./style5" # change accordingly
    # ================================================
    batch_size = 2
    prior_loss_weight = 0.5
    db_eval(instance_prompts, prior_preservation, prior_preservation_class_prompt, 
            num_images, batch_size, prior_loss_weight, prior_preservation_class_folder,
            instance_image_folder, num_samples, output_prompts, output_image_dir)