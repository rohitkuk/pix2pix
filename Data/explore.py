from glob import glob 
import numpy as np
from PIL import Image
import os.path

from common.utils import image_grid

def show_grid(GRIDSIZE, ROWS , COLS ):
    images = glob(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".." , "Dataset/*/*/*/*"))
    print(os.path.join(os.path.abspath(os.path.dirname(__file__)),".." ,"Dataset/*/*/*/*"))
    
    imgs = [Image.open(images[i]) for i in np.random.randint(0, len(images), GRIDSIZE)]
    main_grid = image_grid(imgs, ROWS, COLS)
    main_grid.show()