
from Data import explore, process
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from common.utils import create_dir, check_dir_exists


"""
  ToDos:
    - Data Prepare              : Done
    - Data Explore              : Done
    - Data Process              : Done
    - Data Augmentations        : Done
    - Generator                 : WIP
    - Disriminator              : TBD
    - Training                  : TBD
    - TensorBoard Integration   : TBD
    - WandB Integration         : TBD
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


if __name__ == "__main__":
    explore.show_grid(GRIDSIZE=1, ROWS=1, COLS=1)
    dataset = process.Pix2PixDatasets(dataset_name="edges2shoes")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        create_dir("SampleImages") if not check_dir_exists('SampleImages') else None
        print("here")
        print(x.shape)
        save_image(x, "SampleImages/x.png")
        save_image(y, "SampleImages/y.png")
        import sys
        sys.exit()