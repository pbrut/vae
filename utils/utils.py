"""Data loading and visualization helpers for model/vae.py, kept out of it."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(batch_size=128, data_dir="./data", image_size=64):
    # CelebA images are 178 x 218: crop the central 178 x 178 square (the
    # face), then shrink it to image_size x image_size.
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    train_data = datasets.CelebA(
        root=data_dir, split="train", download=True, transform=transform
    )
    # Unlike MNIST, CelebA is decoded from JPEG files on every access, so
    # load it with worker processes.
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)


def visualize(model, train_loader, device, n=8):
    """Plots a 3 x n grid of images, one row each for:

    1. Inputs: n real CelebA images x from the first batch of train_loader.
    2. Reconstructions: the same images through the full model, i.e.
       x -> encoder -> z ~ q(z|x) -> decoder. Column i reconstructs input i.
    3. Prior samples: z ~ N(0, I) -> decoder, with no encoder and no input
       image, so these are new faces unrelated to the columns above.
    """
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x[:n].to(device) # (n, 3, 64, 64)
        x_hat, _ = model(x) # pixel scale, [0, scaling_factor]
        x_hat = x_hat.reshape(x.shape) / model.config.scaling_factor # back to [0, 1]
        samples = model.generate(n).reshape(x.shape) # [0, 1]

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.6))
    for row, images in enumerate([x, x_hat, samples]):
        for i in range(n):
            # imshow wants RGB images as (height, width, channels).
            axes[row, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[row, i].axis("off")

    # Left-aligned titles on the first column act as row captions (ylabels
    # would be hidden by axis("off")).
    row_captions = [
        r"Inputs: real CelebA images $x$",
        r"Reconstructions: $x \to$ encoder $\to z \sim q_\phi(z \mid x) \to$ decoder",
        r"New images sampled from the model: $z \sim \mathcal{N}(0, I) \to$ decoder",
    ]
    for row, caption in enumerate(row_captions):
        axes[row, 0].set_title(caption, fontsize=10, loc="left")
    # tight_layout can't fit the captions, which are wider than a single image.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, hspace=0.3, wspace=0.05)
    plt.show()
