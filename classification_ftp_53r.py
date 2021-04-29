import os, fnmatch, shutil
from glob import glob
from tqdm import tqdm

SOURCE_PATH = r'D:\ing\FTP'
SOURCE_PATH = SOURCE_PATH.replace('\\', '/')

file_list = os.listdir(SOURCE_PATH)

file_source_path = f'{SOURCE_PATH}/*'
file_path = glob(file_source_path)
file_path_list = [file for file in file_path if file.endswith('.jpg')]

print('Create classification Folder...')
for file in file_list:
    folder_path = f'{file[11:15]}-{file[15:17]}-{file[17:19]}'

    if folder_path == '--':
        continue
    try:
        if not os.path.isdir(folder_path):
            os.mkdir(f'{SOURCE_PATH}/{folder_path}')
    except:
        # print('이미 생성된 폴더가 있습니다.')
        pass

print('Move data files')
for list_ in file_list:
    if len(list_) == 10:
        move_folder = f'{SOURCE_PATH}/{list_}'
        list_ = list_.replace('-', '')
        move_org_file = fnmatch.filter(file_path_list, f'*{list_}*')
        for move_file in tqdm(move_org_file):
            if move_file[-37] == 'O':
                shutil.move(move_file, f'{move_folder}/{move_file[-37:]}')
            if move_file[-36] == 'O':
                shutil.move(move_file, f'{move_folder}/{move_file[-36:]}')

        print(f'{list_}, Done')