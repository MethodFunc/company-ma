import os, fnmatch, cv2

from tqdm import tqdm

TEXT_SOURCE_PATH = r'D:\kids\txt2'
SOURCE_PATH = r'D:\kids\K'
TEXT_SOURCE_PATH = TEXT_SOURCE_PATH.replace('\\', '/')
SOURCE_PATH = SOURCE_PATH.replace('\\', '/')


list_file = fnmatch.filter(os.listdir(TEXT_SOURCE_PATH), '*.txt')
list_file2 = fnmatch.filter(os.listdir(SOURCE_PATH), '*.jpg')

for txt_file, img_file in tqdm(zip(list_file, list_file2)):
    file_path = ''.join([str(TEXT_SOURCE_PATH), "/", str(txt_file)])
    img_file_path = ''.join([str(SOURCE_PATH), "/", str(img_file)])

    img = cv2.imread(img_file_path)
    WEIGHT = img.shape[0]
    HEIGHT = img.shape[1]

    dw = 1 / WEIGHT
    dh = 1 / HEIGHT

    f = open(file_path, 'r', encoding='CP949')
    f2 = open(f'D:/kids/K/{txt_file}', 'w')
    for line in f:
        a = line.rstrip('\n')
        b = a.split(' ')
        b[1] = round(float(b[1]), 0)
        b[2] = round(float(b[2]), 0)
        b[3] = round(float(b[3]), 0)
        b[4] = round(float(b[4]), 0)

        x = (float(b[1]) * 2 + float(b[3])) / 2.0
        y = (float(b[2]) * 2 + float(b[4])) / 2.0
        w = float(b[3])
        h = float(b[4])

        x = round(x * dw, 6)
        w = round(w * dw, 6)
        y = round(y * dh, 6)  # 6자리 표시
        h = round(h * dh, 6)


        if b[0] == 'Boy':
            b[0] = 0

        elif b[0] == 'Girl':
            b[0] = 1

        # print(str(b[0]) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))
        f2.writelines(str(b[0]) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))
        f2.writelines('\n')
    f.close()
    f2.close()
