import torch
from LossFunction import vae_loss, VAEComposedLoss
from DataSets import ShmootDataSet128
from DataLoaders import get_shmoot_dataloader
from VAEModel import Vae
from Utils import visualize_batch, make_image_grid
from torch.utils.tensorboard import SummaryWriter
import time


TB_RUN_NAME = 'Medium_NoBatchnorm'
# --- hyper parameters ---
BATCH_SIZE = 16
EPOCHS = 12000
LEARNING_RATE = 1e-4
LR_DECAY_GAMMA = 0.1
LATENT_SIZE = 256
RESIDUAL_BLOCKS = 3
FC_LAYERS = [512, 256]
BETA = 1.5
VGG_WEIGHT = 0.005
BATCHNORM = False ####################################################

NUM_WORKERS = 4
EVAL_FREQ = 200
CHECKPOINT_FREQ = 1500


def main():
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)

    data_set = ShmootDataSet128(augment=True)
    data_loader = get_shmoot_dataloader(data_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = Vae(latent_size=LATENT_SIZE,
                num_residual_convs=RESIDUAL_BLOCKS,
                fc_layers=FC_LAYERS,
                batchnorm=BATCHNORM,
                device=device).to(device)
    # model = torch.load('checkpoints/withVGG7500.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2*EPOCHS//3, gamma=LR_DECAY_GAMMA)
    loss_fn = VAEComposedLoss(kl_beta=BETA, vgg_weight=VGG_WEIGHT, device=device)

    writer = SummaryWriter(comment=TB_RUN_NAME)

    for epoch in range(EPOCHS+1):
        start = time.time()
        for batch_idx, im_batch in enumerate(data_loader):
            im_batch = im_batch.to(device)
            recon_batch, z_mu, z_sigma, z = model(im_batch)

            loss, vgg_loss, recon_mse_loss, latent_kl = loss_fn(im_batch, recon_batch, z_mu, z_sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/0loss", loss, iteration)
            writer.add_scalar("Loss/recon_error", recon_mse_loss, iteration)
            writer.add_scalar("Loss/VGG_loss", vgg_loss, iteration)
            writer.add_scalar("Loss/latent_kl", latent_kl, iteration)

        print("Epoch: {}/{}".format(epoch, EPOCHS), "Time: {:.2f}".format(time.time() - start))
        if epoch % CHECKPOINT_FREQ == 0:
            torch.save(model, "checkpoints/{}{}.pt".format(TB_RUN_NAME, epoch))
        if epoch % EVAL_FREQ == 0:
            writer.add_image("Images/original_subset", make_image_grid(im_batch[0:8]), epoch)
            writer.add_image("Images/reconstructed_subset", make_image_grid(recon_batch[0:8]), epoch)
            writer.add_image("Images/sampled_images", make_image_grid(model.sample(8)), epoch)

        scheduler.step()
        writer.add_scalar("Utils/learning_rate", scheduler.get_last_lr()[-1], epoch)


if __name__ == '__main__':
    main()
