import torch
from LossFunction import vae_loss
from DataSets import ShmootDataSet128
from DataLoaders import get_shmoot_dataloader
from VAEModel import Vae
from Utils import visualize_batch
from torch.utils.tensorboard import SummaryWriter
import time

# --- hyper parameters ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
LATENT_SIZE = 128


def main():
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    writer = SummaryWriter()

    data_set = ShmootDataSet128(augment=False)
    data_loader = get_shmoot_dataloader(data_set, batch_size=BATCH_SIZE)
    model = Vae(latent_size=LATENT_SIZE, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        start = time.time()
        for batch_idx, im_batch in enumerate(data_loader):
            im_batch = im_batch.to(device)
            recon_batch, z_mu, z_sigma, z = model(im_batch)

            loss, recon_error, latent_kl = vae_loss(im_batch, recon_batch, z_mu, z_sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/0loss", loss, epoch * len(data_loader) + batch_idx)
            writer.add_scalar("Loss/recon_error", recon_error, epoch * len(data_loader) + batch_idx)
            writer.add_scalar("Loss/latent_kl", latent_kl, epoch * len(data_loader) + batch_idx)

        print("Epoch: {}/{}".format(epoch, EPOCHS), "Time: {:.2f}".format(time.time() - start))
        if epoch % 5 == 0:
            visualize_batch(recon_batch[0:8])
            visualize_batch(model.sample(12))


if __name__ == '__main__':
    main()
