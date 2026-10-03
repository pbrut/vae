import torch

from utils.utils import get_dataloader, visualize
from model.vae import VAE, VAEConfig

epochs = 100
batch_size = 256
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
train_loader = get_dataloader(batch_size=batch_size)

config = VAEConfig()
model = VAE(config=config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        _, loss = model(x)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader.dataset):.4f}")

torch.save(model.state_dict(), "model/vae.pt")
print("Saved model to model/vae.pt")

visualize(model, train_loader, device)
