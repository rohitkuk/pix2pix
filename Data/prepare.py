from common.utils import extractData ,kaggleDownloadData
from common.utils import check_dir_exists, create_dir
from common.utils import remove_files, AlreadyDownloaded



def main(Data_Path, expectedFileName = False, unzip_path = "Dataset", keep_cache = True):
    """ 
        This functions download kaggle dataset,
        extract the files and remove if required
    """
    print('1')
    # IF EXPECTED FILE NAME NOT GIVEN TRY MAKING ONE WITH THE DATA PATH
    if not expectedFileName:
        expectedFileName = Data_Path.split('/')[-1] + '.zip'
    print('2')
    # CHECK IF ALREADY DOWNLOADED
    if AlreadyDownloaded(expectedFileName):
        return
    
    print("3")
    # DOWNLOAD KAGGLE DATASET WOULD NEED PATH TO KAGGLE JSON
    kaggleDownloadData(Data_Path)
    
    # CREATE UNZIP DIRECTORY IF NOT ALREADY PRESENT 
    if not check_dir_exists(dir_=unzip_path):
        create_dir(unzip_path)
    
    # EXTRACT THE DOWNLOADED DATA TO UNZIP FOLDER
    extractData(filepath=expectedFileName,unzip_path=unzip_path)
    
    # REMOVE THE UNZIPPED FILE
    if not keep_cache:
        remove_files(filepath = expectedFileName, match = False, match_string = None)


if __name__ == '__main__':
    main(
        Data_Path="vikramtiwari/pix2pix-dataset",
        expectedFileName=False,
        unzip_path= "Dataset",
        keep_cache=False
        )



