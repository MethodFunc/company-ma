import os, fnmatch

source_path = r'E:\MK-SD53R'
source_path = source_path.replace('\\', '/')

folder_list = os.listdir(source_path)

folder_path_list = []
for folder in folder_list:
    path = f'{source_path}/{folder}'

    folder_path_list.append(path)

data_201 = []
data_202 = []
try:
    for folder in folder_path_list:
        path = f'{folder}/201'
        confirm = fnmatch.filter(os.listdir(path), '*.jpg')

        if len(confirm) == 0:
            pass
        else:
            pp_path = folder.split('/')
            data_201.append(pp_path[2])

        path = f'{folder}/202'
        confirm = fnmatch.filter(os.listdir(path), '*.jpg')

        if len(confirm) == 0:
            pass
        else:
            pp_path = folder.split('/')
            data_202.append(pp_path[2])
except Exception as err:
    pass


for data in data_201:
    print(f'미 분류 201: {data}')


for data in data_202:
    print(f'미 분류 202: {data}')