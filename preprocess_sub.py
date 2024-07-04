import os
import zipfile
import logging


def exec_process(pm):
    logging.info('[hunmin log] the start line of the function [exec_process]')

    # 저장 파일 확인
    list_files_directories(pm.source_path)
    
    # pm.source_path의 dataset.zip 파일을
    # pm.target_path 경로에 압축해제
    my_zip_path = os.path.join(pm.source_path,'meta_data.zip')
    extract_zip_file = zipfile.ZipFile(my_zip_path)
    extract_zip_file.extractall(pm.source_path)
    extract_zip_file.close()
    
    # 저장 파일 확인
    list_files_directories(pm.target_path)

    logging.info('[hunmin log] the finish line of the function [exec_process]')


# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))  