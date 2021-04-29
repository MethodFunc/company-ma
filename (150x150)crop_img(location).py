import glob
import cv2
import os
from PIL import Image
from roi import setting_roi
# surface = ['snow']  # surface list change
path = r'F:\MK-SD53R\2021-01-08\202\normal_day'  # path change
save_path = r'D:\Harry\002.Working\Normal_ROI_CROP'  # save_path list change
path = path.replace('\\', '/')
save_path = save_path.replace('\\', '/')

images = glob.glob(f'{path}/*.jpg')  # {surface[0]}

ROI = [(1, 10), (2, 8), (2, 9), (2, 10), (3, 7), (3, 8), (3, 9), (4, 7), (4, 10), (4, 11), (5, 9), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10)]

ROI_WIDTH, ROI_HEIGHT = 150, 150
count = 0

print("image crop start")
print(f'image_path : {path}')
print(f'save_path : {save_path}')
for img in images:
    filedir = os.path.basename(img)
    filename = os.path.splitext(filedir)
    # print(filename)
    img = cv2.imread(img)
    roi_set = []

    for (i, j) in ROI:
        x, y = i * ROI_WIDTH, j * ROI_HEIGHT
        roi_img = img[y:y + ROI_HEIGHT, x:x + ROI_WIDTH]
        # print(roi_img)
        # roi_set.append(roi_img)
        cv2.imwrite(f'{save_path}/snow_{count}_0.jpg', roi_img)
        count += 1
print(f'count : {count}')
print('-----The end-----')
