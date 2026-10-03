# Variational Autoencoder

A step-by-step derivation and mathematical justification of Evidence Lower Bound (ELBO) maximization in the original Variational Autoencoder ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)), together with a convolutional VAE trained on CelebA.

## Contents

The derivation lives in [`derivation.ipynb`](derivation.ipynb):

1. **Pre-requisites**: convexity and concavity, Jensen's inequality, KL divergence
2. **Problem statement**: maximum likelihood, why a regular autoencoder is not a generative model, and how a VAE fixes that
3. **Evidence Lower Bound (ELBO)**: deriving the bound with Jensen's inequality, and why maximizing it also pulls the approximate posterior towards the true one
4. **Reparameterization trick**: predicting the log-variance, and rewriting sampling as $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ so gradients can flow from the decoder back into the encoder
5. **Loss function**: the closed-form KL term, a single-sample Monte Carlo estimate of the reconstruction term, and Binary Cross-Entropy as the reconstruction loss

## Code

| File | Purpose |
|---|---|
| `model/vae.py` | Convolutional VAE, plus a simpler fully connected `SimpleVAE` |
| `train.py` | Trains the VAE on CelebA and saves the weights to `model/vae.pt` |
| `generate.py` | Samples new faces from the prior, $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |

## Usage

```bash
uv sync                      # install dependencies
uv run train.py              # train on CelebA
uv run generate.py 64        # generate 64 new faces
```

CelebA is downloaded from Google Drive by torchvision, which sometimes fails with a "quota exceeded" error. If that happens, download the dataset manually into `data/celeba/`.
