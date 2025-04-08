import os
import re
import torch
from PIL import ImageDraw, ImageFont
from datetime import datetime
from diffusers import StableDiffusionPipeline

# ===== Configuration =====
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

model_id = "runwayml/stable-diffusion-v1-5"
prompt = "A cozy coffee shop interior during sunset"
signature_text = "@ualvi27"
font_size = 24
guidance_scale = 7.5
steps = 40

# ===== Load Model =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# ===== Generate Image =====
print("Generating image for prompt:", prompt)
image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]

# ===== Add Signature =====
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    font = ImageFont.load_default()

position = (image.width - 250, image.height - 40)
draw.text(position, signature_text, font=font, fill=(255, 255, 255))

# ===== Save Image =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = re.sub(r'[\\/*?:"<>|]', "", prompt)[:25].replace(" ", "_") + f"_{timestamp}.png"
filepath = os.path.join(output_dir, filename)
image.save(filepath)

print("✅ Image saved at:", filepath)
