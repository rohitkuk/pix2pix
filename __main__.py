
from Data import explore, process, prepare
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from common.utils import create_dir, check_dir_exists
import Config
from Execute.train import train
import torch.nn as nn 
from torch import optim
from Models.generator import Generator
from Models.discriminator import Discriminator
from rich import pretty

pretty.install()

"""
  ToDos:
    - Data Prepare              : Done
    - Data Explore              : Done
    - Data Process              : Done
    - Data Augmentations        : Done
    - Generator                 : Done
    - Disriminator              : Done
    - Training                  : Done
    - TensorBoard Integration   : TBD
    - WandB Integration         : TBD̦
    - Logging                   : TBD (Try Implementing Rich Library)
    - Argument Parsing          : TBD
    - Requirments.txt           : TBD
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

    disc_optim = optim.Adam(disc.parameters(), lr = Config.LEARNING_RATE, betas=(0.5, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr = Config.LEARNING_RATE, betas=(0.5, 0.999))

    print("Start Training")
    
    for epoch in range(Config.EPOCHS):
        train(disc, gen , BCE, disc_optim, gen_optim, l1_loss, epoch, loader, Config)





if __name__ == "__main__":
    # explore.show_grid(GRIDSIZE=1, ROWS=1, COLS=1)
    # for x, y in loader:
        # create_dir("SampleImages") if not check_dir_exists('SampleImages') else None
        # print("here")
        # print(x.shape)
        # save_image(x, "SampleImages/x.png")
        # save_image(y, "SampleImages/y.png")
        # import sys
        # sys.exit()
    main(dataset_name = "edges2shoes", Data_Path="vikramtiwari/pix2pix-dataset")