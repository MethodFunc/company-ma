import os, fnmatch, cv2, shutil, random
import pandas as pd
from pprint import pprint
import numpy as np
from tqdm import tqdm


class bright_calc:
    def __init__(self, FOLDER_PATH, DATE, SOURCE_PATH, ROI):
        self.folder_path = FOLDER_PATH.replace('\\', '/')
        self.date = DATE
        self.source_path = SOURCE_PATH.replace('\\', '/')
        self.roi = ROI

        self.roi_width, self.roi_height = 150, 150

    def __call__(self):
        # self.extraction_image()
        self.brightness_calc()

    def extraction_image(self):
        folder_list = fnmatch.filter(os.listdir(self.folder_path), self.date)

        if not os.path.isdir(self.source_path):
            os.mkdir(self.source_path)

        pprint('Image Copy...')
        for folder in tqdm(folder_list):
            folder_path_2 = ''.join([str(self.folder_path), "/", str(folder), "/201/dry_day"])

            if not os.path.isdir(folder_path_2):
                continue
            file_list_12 = fnmatch.filter(os.listdir(folder_path_2), f'*{folder[-2:]}12*.jpg')
            file_list_13 = fnmatch.filter(os.listdir(folder_path_2), f'*{folder[-2:]}13*.jpg')
            file_list_14 = fnmatch.filter(os.listdir(folder_path_2), f'*{folder[-2:]}14*.jpg')

            for file in file_list_12:
                file_path = ''.join([str(folder_path_2), "/", str(file)])
                shutil.copy(f'{file_path}', f'{self.source_path}/{file}')

            for file in file_list_13:
                file_path = ''.join([str(folder_path_2), "/", str(file)])
                shutil.copy(f'{file_path}', f'{self.source_path}/{file}')

            for file in file_list_14:
                file_path = ''.join([str(folder_path_2), "/", str(file)])
                shutil.copy(f'{file_path}', f'{self.source_path}/{file}')

        pprint('Image Copy Done.')

    def brightness_calc(self):
        avg_val = []
        items = []

        path = ''.join([str(self.source_path).replace("\\", "/"), "/"])
        image_file_list = fnmatch.filter(os.listdir(path), '*.jpg')
        random.shuffle(image_file_list)

        img_file_list = fnmatch.filter(os.listdir(self.source_path), '*.jpg')
        pprint("Processing Image...")
        for img_file in tqdm(img_file_list):
            try:
                img_file_path = ''.join([str(path), "/", str(img_file)])
                img = cv2.imread(img_file_path)
                for (i, j) in self.roi:
                    x, y = i * self.roi_width, j * self.roi_height
                    roi_img = img[y:y + self.roi_height, x:x + self.roi_width]
                    items.append(roi_img)
            except Exception as err:
                print(f'image processing failed at {str(img_file)}')
                pass
        pprint("Processing Image Done.")

        pprint("Processing bright image...")
        for img_file in tqdm(items):
            lab = cv2.cvtColor(img_file, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            y, x, z = img_file.shape
            l_blur = cv2.GaussianBlur(l, (11, 11), 5)
            maxval = []

            count_percent = 5
            count_percent = count_percent / 100
            row_percent = int(count_percent * x)
            column_percent = int(count_percent * y)

            for i in range(1, x - 1):
                if i % row_percent == 0:
                    for j in range(1, y - 1):
                        if j % column_percent == 0:
                            pix_cord = (i, j)

                            # cv2.circle(img_dot, (int(i), int(j)), 5, (0, 255, 0), 2)
                            img_segment = l_blur[i:i + 3, j:j + 3]
                            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                            maxval.append(maxVal)

            avg_maxval = round(sum(maxval) / len(maxval))
            avg_val.append(avg_maxval)
        avg_val = np.array(avg_val).reshape(-1, len(roi))
        df = pd.DataFrame(avg_val, columns=['roi1', 'roi2', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7', 'roi8'])
        pprint("Save dataframe to csv")
        df.to_csv(f'{self.source_path[-2:]}_bright.csv', index=False)
        pprint("Processing bright Done.")


if '__main__' == __name__:
    roi = [(0, 10), (0, 11), (1, 9), (1, 10), (1, 15), (3, 13), (5, 10), (5, 11)]
    folder_path = 'E:/MK-SD53R'
    date = '2021-01-*'
    source_path = f'D:/Harry/Bright/MK-SD53R/{date[:7]}'

    start = bright_calc(FOLDER_PATH=folder_path, DATE=date, SOURCE_PATH=source_path, ROI=roi)

    start()