
# tree --dirsfirst --noreport -I 'Dataset*|wandb*|__pycache__|__init__.py|logs|SampleImages|List.md' > List.md 

from Data import explore, process, prepare
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from common.utils import create_dir, check_dir_exists, initialize_weights
import Config
from Execute.train import train
import torch.nn as nn 
from torch import optim
from Models.generator import Generator
from Models.discriminator import Discriminator
from rich import pretty
import sys
# import os

pretty.install()
import wandb

wandb.init(project="pix2pix", entity="rohitkuk")
"""
  ToDos:
    - Data Prepare              : Done
    - Data Explore              : Done
    - Data Process              : Done
    - Data Augmentations        : Done
    - Generator                 : Done
    - Disriminator              : Done
    - Training                  : Done
    - Requirments.txt           : Done
    - TensorBoard Integration   : Done
    - WandB Integration         : Done
    - ReadME.md                 : WIP
    - Logging                   : TBD (Try Implementing Rich Library)
    - Argument Parsing          : TBD
    - Packaging                 : TBD
    - Frontend or UI            : TBD
    - Optimization              : Continous
    - Test
        - Unit                  : TBD
        - Integration           : TBD
        - System                : TBD
        - UAT                   : TBD
"""



def main(dataset_name, Data_Path):
    
    print("Preparing Dataset")
    prepare.main( Data_Path, expectedFileName=False, unzip_path= "Dataset", keep_cache=False)
    dataset = process.Pix2PixDatasets(dataset_name)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers= Config.NUM_WORKERS  if Config.NUM_WORKER else 0)
    
    BCE = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    disc = Discriminator(Config.IMG_CHANNELS, Config.FEATURES).to(Config.DEVICE)
    gen  = Generator(Config.IMG_CHANNELS, Config.FEATURES).to(Config.DEVICE)

    initialize_weights(disc), initialize_weights(gen)
    disc_optim = optim.Adam(disc.parameters(), lr = Config.LEARNING_RATE, betas=(0.5, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr = Config.LEARNING_RATE, betas=(0.5, 0.999))

    # Tensorboard Implementation
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")

    wandb.watch(gen)
    wandb.watch(disc)

    # training
    disc.train()
    gen.train()
    step = 0

    print("Start Training")
    for epoch in range(Config.EPOCHS):
        step = train(disc, gen, BCE, disc_optim, gen_optim, l1_loss,epoch,loader, Config , make_grid, wandb, writer_real, writer_fake, step)




if __name__ == "__main__":
    main(dataset_name = sys.argv[1], Data_Path="vikramtiwari/pix2pix-dataset")
    # explore.show_grid(GRIDSIZE=1, ROWS=1, COLS=1)
    # for x, y in loader:
        # create_dir("SampleImages") if not check_dir_exists('SampleImages') else None
        # print("here")
        # print(x.shape)
        # save_image(x, "SampleImages/x.png")
        # save_image(y, "SampleImages/y.png")
        # import sys
        # sys.exit()