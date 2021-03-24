from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
import glob, os, multiprocessing
from PIL import Image


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
    return os.path.isfile(expectedFileName)



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