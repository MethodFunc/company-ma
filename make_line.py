import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as im

# filepath = 'D:/33A,33C_data/33A/0930'

filepath = r'D:\Harry\000.DataAnalysis\002.python\ROI_IMAGE'
filepath = filepath.replace('\\', '/')

save_folder = r'D:\Harry\000.DataAnalysis\003.ROI_image'
save_folder = save_folder.replace('\\', '/')

filelist = []
try:
    for filename in os.listdir(filepath):
        if filename.endswith('.JPG'):
            filelist.append(filename)
except Exception as err:
    print(err)
    filelist.append(filepath)

print(f'filelist len: {len(filelist)}')

for i in range(len(filelist)):
    img = cv2.imread(f'{filepath}/{filelist[i]}', cv2.IMREAD_ANYCOLOR)
    img_width, img_height = img.shape[0], img.shape[1]
    print(f'img_width: {img_width}\t\timg_height: {img_height}')

    line_width = 0
    while True:
        # img = cv2.line(img, (line_width, 0), (line_width, img_height), (100, 100, 100), 2)
        img = cv2.line(img, (0, line_width), (img_height, line_width), (0, 255, 255), 2)
        line_width += 150
        if line_width >= img_width:
            break

    line_height = 0
    while True:
        # img = cv2.line(img, (0, line_height), (img_width, line_height), (100, 100, 100), 2)
        img = cv2.line(img, (line_height, 0), (line_height, img_width), (0, 255, 255), 2)
        line_height += 150
        if line_height >= img_height:
            break

    # cv2.imshow("test", img)
    cv2.imwrite(f'{save_folder}/2ROI150_{filelist[i]}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# for i in range(len(filelist)):
#     fig = plt.figure(figsize=(7,10))
#     plt_img = im.imread(f'{save_folder}/ROI150_{filelist[0]}')
#     plt.imshow(plt_img)
#     plt.show()