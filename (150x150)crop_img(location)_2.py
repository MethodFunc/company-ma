import glob
import cv2
import os
import random
from tqdm import tqdm

# surface = ['snow']  # surface list change
path = r'D:\road\road_train_201_203' # path change

path = path.replace('\\', '/')
save_path = rf'{path}/crop'  # save_path list change
save_path = save_path.replace('\\', '/')


if not os.path.isdir(f'{save_path}'):
    os.mkdir(f'{save_path}')

categories = ['dry_day', 'dry_night', 'wet_day', 'wet_night']
ROI_WIDTH, ROI_HEIGHT = 150, 150
# ROI = [(1, 14), (1, 15), (2, 14), (2, 15), (3, 14), (3, 15)]
ROI = [(0, 9), (0, 10), (0, 11), (0, 12), (1, 8), (1, 9), (1, 10), (1, 14), (1, 15), (2, 7), (2, 8), (2, 12), (2, 13), (2, 14), (2, 15), (3, 6), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 4), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (5, 3), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 5), (6, 6), (6, 7), (6, 8), (7, 3), (7, 4)]
#
# ROI = [(2, 14), (3, 14), (3, 15), (13, 15), (13, 16), (14, 15), (14, 16), (23, 14), (24, 14), (24, 15)]
images_number = 300

print("image crop start")
print(f'image_path : {path}')
print(f'save_path : {save_path}')
print(f'Roi_Image_number : {len(ROI) * images_number}')
for cat in categories:
    dir_path = f'{path}/{cat}'

    if not os.path.isdir(dir_path):
        print('Path error')
        exit()

    if not os.path.isdir(f'{save_path}/{cat}'):
        os.mkdir(f'{save_path}/{cat}')

    img_list = glob.glob(f'{dir_path}/*.jpg')
    random.shuffle(img_list)
    img_list = img_list[:images_number]

    count = 0

    for img in tqdm(img_list):
        img = cv2.imread(img)
        roi_set = []
        for (i, j) in ROI:
            x, y = i * ROI_WIDTH, j * ROI_HEIGHT
            roi_img = img[y:y + ROI_HEIGHT]
            cv2.imwrite(f'{save_path}/{cat}/{cat}_{count}.jpg', roi_img)
            count += 1
print(f'count : {count}')
print('-----The end-----')
