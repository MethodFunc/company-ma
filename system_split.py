import os
import fnmatch
import shutil


def system_splits(paths):
    folder_list = os.listdir(paths)
    system_folder = [201, 202, 203]

    for folder in folder_list:
        main_path = f"{paths}/{folder}"
        if not os.path.isdir(main_path):
            continue

        file_list = fnmatch.filter(os.listdir(main_path), "*.jpg")

        if len(file_list) == 0:
            print(f"{folder} 폴더 안에 이미지 파일이 없습니다.")
            continue

        print(f"{folder} start...")

        for system in system_folder:
            if not os.path.isdir(f"{main_path}/{system}"):
                os.mkdir(f"{main_path}/{system}")

        for file in file_list:
            file_path = f"{main_path}/{file}"

            for system in system_folder:
                if f"OPTICAL{system}" in file:
                    shutil.move(file_path, f"{main_path}/{system}/{file}")

        print(f"{folder} End..")


if __name__ == "__main__":
    path = '/Volumes/Macintosh HD/Users/methodfunc/Pictures/MK-SD53R'
    system_splits(paths=path)
