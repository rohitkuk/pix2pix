from os.path import join
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
import glob, os, multiprocessing
from PIL import Image
import albumentations as A
import torch.nn as nn
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    from albumentations.pytorch import ToTensor as ToTensorV2

from tqdm import tqdm

def kaggleDownloadData(Data_Path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(Data_Path, quiet=False)


def extractData(filepath, unzip_path):
    print("Extracting...")
    with ZipFile(file=filepath) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=unzip_path)

    print("File Unzipped Succesfully...")
    


def remove_files(filepath, match = False, match_string = None):
    if match:
        if match_string==None:print( "Match Strin Required if match=True") #Raise an Exception
        p = multiprocessing.Pool(4)
        p.map(os.remove, glob.glob(match_string))
        print("Files Succesfully Removed") #Add print log instead
    if not match:
        os.remove(filepath)
        print("File Succesfully Removed")


def check_dir_exists(dir_):
    return os.path.isdir(dir_)


def create_dir(dir_):
    os.mkdir(dir_)


def AlreadyDownloaded(expectedFileName):
    if check_dir_exists("Dataset"):
        print("Directory Existed")
        result = os.path.isfile(expectedFileName) or len(os.listdir("Dataset")) != 1
        print(result)
        return result
    print("DIRECTORY NOT EXISTED")
    result = os.path.isfile(expectedFileName)
    return result


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def dataset_root(dataset_name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), ".." , f"Dataset/{dataset_name}/{dataset_name}/train")


def list_files(dir_):
    return [os.path.join(dir_, i) for i in os.listdir(dir_)]


def image_splitter(image, dataset_name, reverse = False):
    reverse = not reverse if dataset_name == 'edges2shoes' else reverse
    input_image = image[:,int(image.shape[1]/2):, :]
    target_image = image[:,:int(image.shape[1]/2), :]
    if reverse:
        input_image, target_image = target_image, input_image
    return input_image, target_image


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

input_augmentation= A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
])

target_augmentation= A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)