import os, fnmatch, shutil
from tqdm import tqdm
from pprint import pprint

SOURCE_PATH = r'D:\kids\K\done'
SOURCE_PATH = SOURCE_PATH.replace('\\', '/')

file_list = fnmatch.filter(os.listdir(SOURCE_PATH), '*.txt')

for file in tqdm(file_list):
    path = ''.join([str(SOURCE_PATH), "/", str(file)])

    if not os.path.isdir(f'{SOURCE_PATH}/Complete'):
        os.mkdir(f'{SOURCE_PATH}/Complete')

    try:
        if os.path.getsize(path) != 0:
            if file == 'classes.txt':
                shutil.move(path, f'{SOURCE_PATH}/Complete/{file}')
                pass
            shutil.move(f'{path[:-4]}.jpg', f'{SOURCE_PATH}/Complete/{file[:-4]}.jpg')
            shutil.move(path, f'{SOURCE_PATH}/Complete/{file}')

    except Exception as err:
        pprint(err)
        pass
