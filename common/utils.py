from os.path import join
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
import glob, os, multiprocessing
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def kaggleDownloadData(Data_Path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(Data_Path)


def extractData(filepath, unzip_path):
    print("Extracting...")
    with ZipFile(filepath, "r") as f:
        f.extractall(unzip_path)
    print("File Unzipped Succesfully...", end = "\r")
    


def remove_files(filepath, match = False, match_string = None):
    if match:
        if match_string==None:print( "Match Strin Required if match=True") #Raise an Exception
        p = multiprocessing.Pool(4)
        p.map(os.remove, glob.glob(match_string))
        print("Files Succesfully Removed") #Add print log instead
    if not match:
        os.remove(filepath)
        print("File Succesfully Removed")



def AlreadyDownloaded(expectedFileName):
    return os.path.isfile(expectedFileName) and len(os.listdir("Dataset")) != 0



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid



def check_dir_exists(dir_):
    return os.path.isdir(dir_)


def create_dir(dir_):
    os.mkdir(dir_)


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