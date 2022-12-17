from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPTokenizer
model_id = "/mnt/outputs" # where finetuned model checkpoint is saved
#model_id = "runwayml/stable-diffusion-v1-5" # baseline
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
pipe = StableDiffusionPipeline.from_pretrained(model_id, 
tokenizer = tokenizer,
torch_dtype=torch.float16, safety_checker = None).to("cuda")


objects =  ["pizza", "peach", "phone", "laptop", "umbrella", "cupcake", "pancake"]
#objects =  ["apple", "avocado", "banana", "beer Bottle", "beer Glass", "beer", "beet", "bread", "broccoli"]

num_added_tokens = tokenizer.add_tokens("<custom-style>")
assert num_added_tokens == 0

# change accordingly
output_folder = "customUNetTI/style9/repeats20steps100/" 
for obj in objects:
    prompt = "A {} in the style of <custom-style>".format(obj)
    image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
    image.save(output_folder+("_").join(str(prompt).split())+".png")