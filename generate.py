import argparse
import math

import matplotlib.pyplot as plt
import torch

from model.vae import VAE, VAEConfig

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, help="number of images to generate")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = VAEConfig()
model = VAE(config=config).to(device)
model.load_state_dict(torch.load("model/vae.pt", map_location=device))

model.eval()
with torch.no_grad():
    images = model.generate(args.n)
    images = images.reshape(-1, config.image_channels, config.image_size, config.image_size)
    images = images.permute(0, 2, 3, 1).cpu() # (n, 64, 64, 3), the layout imshow wants

cols = min(args.n, 8)
rows = math.ceil(args.n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, image in zip(axes.flat, images):
    ax.imshow(image)
plt.tight_layout()
plt.show()
