import os, fnmatch, random, shutil
from tqdm import tqdm


def random_sampling(source_path, save_path, test_set_number, type1 = None):

    folder_list = os.listdir(source_path)

    for folder in folder_list:
        if type1 is None:
            path = f'{source_path}/{folder}'
            save_paths = f"{save_path}/{path.split('/')[-1]}"
        else:
            path = f"{source_path}/{type1}/{folder}"
            save_paths = f"{save_path}/{type1}/{path.split('/')[-1]}"
        try:
            files = fnmatch.filter(os.listdir(path), '*.jpg')

            if not files:
                print(f"{path}의 이미지가 없습니다. 경로를 다시 설정해주세요.")
                break
            if len(files) < test_set_number:
                print(f'{path}의 샘플이 설정된 {test_set_number}보다 부족하여 패스합니다.')
                break

            files = random.sample(files, test_set_number)

            for file in tqdm(files):
                file_path = f'{path}/{file}'

                if not os.path.isdir(save_paths):
                    os.mkdir(save_paths)

                shutil.move(f'{file_path}', f'{save_paths}/{file}')
        except:
            pass


if __name__ == '__main__':
    source_path = "/Users/methodfunc/Pictures/wet&moist"
    save_path = "/Users/methodfunc/Pictures/wet&moist"
    test_set_number = 2
    # type1 = 203
    # type2 = 'normal'

    random_sampling(source_path=source_path, save_path=save_path, test_set_number=test_set_number)
