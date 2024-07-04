import os
import shutil
import tempfile
import base64
import pandas as pd
import logging
import ipywidgets
from ipywidgets import FileUpload
from IPython.display import FileLink
from PIL import Image

ROOT = os.path.dirname(__file__)

# t3qai_client 클래스: t3qai_client 객체
class t3qai_client:
    def train_start(self):
        return None

    def train_finish(self, result, result_msg):
        if result_msg != "success":
            raise Exception(result_msg)
        else:
            logging.info(result)
            logging.info("train finish")

    def train_load_param(self):
        '''set_param'''
        epoch = 20
        batch_size = 16
        params = {"epoch" : epoch, 'batch_size' : batch_size}
        return { **params }

class PM:
    def __init__(self):
        self.source_path = f'{ROOT}'
        self.target_path = f'{ROOT}/meta_data'

class UploadFile:
    def __init__(self, file, filename):
        self.file = file
        self.filename = filename

def DownloadFile(file_name, file_obj = None, file_path = None):
    file_route = './meta_data/DownloadFiles'
    os.makedirs(file_route, exist_ok = True)
    file_dir = os.path.join(file_route, file_name)
    if (file_obj == None) == (file_path == None):
        Err_msg = "[DownloadFile Error]: Only one of the 'file_path' or 'file_obj' arguments is required."
        Err_msg += f"{0 if file_obj==None else 2} arguments entered."
        raise Exception(Err_msg)
    elif(file_obj != None):
        file_obj.seek(0)
        file_read = base64.b64encode(file_obj.read()).decode('utf-8')
        binary_file = base64.b64decode(file_read)
        with open(file_dir, 'wb') as f:
            f.write(binary_file)
    elif(file_path != None):
        shutil.copyfile(file_path, file_dir)

    return FileLink(file_dir)

pm = PM()

T3QAI_TRAIN_OUTPUT_PATH = f'{ROOT}/meta_data'
T3QAI_TRAIN_MODEL_PATH =  f'{ROOT}/meta_data'
T3QAI_TRAIN_DATA_PATH =  f'{ROOT}/meta_data'
T3QAI_TEST_DATA_PATH =  f'{ROOT}/meta_data'
T3QAI_MODULE_PATH =  f'{ROOT}/meta_data'
T3QAI_INIT_MODEL_PATH =  f'{ROOT}/meta_data'

# t3qai_client 객체
tc = t3qai_client()
print('T3QAI_TRAIN_OUTPUT_PATH:', T3QAI_TRAIN_OUTPUT_PATH)
print('T3QAI_TRAIN_MODEL_PATH:', T3QAI_TRAIN_MODEL_PATH)
print('T3QAI_TRAIN_DATA_PATH:', T3QAI_TRAIN_DATA_PATH)
print('T3QAI_TEST_DATA_PATH:', T3QAI_TEST_DATA_PATH)
print('T3QAI_MODULE_PATH:', T3QAI_MODULE_PATH)
print('T3QAI_INIT_MODEL_PATH:', T3QAI_INIT_MODEL_PATH)

# init_svc(im, rule) 함수 입력
im = None
rule = None
# transform(df, params, batch_id) 함수 입력
batch_id = 0

import io
import pandas as pd

# base64 encoded image - 00001.jpeg 
image = Image.open('./meta_data/dataset/test/00001.jpeg')
buffered = io.BytesIO()
image.save(buffered, format="PNG")
base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
image_data = [[base64_image]]
df = pd.DataFrame(image_data)

# inference_file 함수 추론
files = []

uploader = FileUpload(accept='*', multiple=True, description='select data', button_style='danger')
def uploader_change(change):
    uploader.button_style='success'
    count = len(uploader.value)
    uploader._counter = count
    files.clear()
    for file_num in range(count):
        temp_data = tempfile.TemporaryFile()
        if ipywidgets.__version__[0] == '7':
            temp_data.write(list(uploader.value.values())[file_num]['content'])
            file = UploadFile(temp_data, pd.DataFrame(list(uploader.value.values())[file_num]).iloc[1,0])
        elif int(ipywidgets.__version__[0]) > 7:
            temp_data.write(uploader.value[file_num].content)
            file = UploadFile(temp_data, uploader.value[file_num].name)
        files.append(file)

uploader.observe(uploader_change, 'value')