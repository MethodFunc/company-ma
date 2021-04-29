import os, fnmatch, random, shutil
from tqdm import tqdm


def random_sampling(source_path, save_path, test_set_number):
    source_path = source_path.replace('\\', '/')
    save_path = save_path.replace('\\', '/')

    folder_list = os.listdir(source_path)

    for folder in folder_list:
        path = f'{source_path}/{folder}'
        save_paths = f"{save_path}/{path.split('/')[-1]}"
        try:
            files = fnmatch.filter(os.listdir(path), '*.jpg')
            if len(files) < test_set_number:
                print(f'{path}의 샘플이 설정된 {test_set_number}보다 부족하여 패스합니다.')
                pass
            random.shuffle(files)
            files = random.sample(files, test_set_number)

            for file in tqdm(files):
                file_path = f'{path}/{file}'

                if not os.path.isdir(save_paths):
                    os.mkdir(save_paths)

                shutil.move(f'{file_path}', f'{save_paths}/{file}')
        except:
            pass


if __name__ == '__main__':
    source_path = r'D:\road\slush_test\train'
    save_path = r"D:\road\slush_test\test"
    test_set_number = 400
    type1 = 203
    type2 = 'normal'

    random_sampling(source_path=source_path, save_path=save_path, test_set_number=test_set_number)
