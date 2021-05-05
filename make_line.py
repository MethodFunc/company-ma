import cv2
import os


def make_line(source_path, save_path, width, height):
    file_list = []
    try:
        for filename in os.listdir(source_path):
            if filename.endswith('.JPG'):
                file_list.append(filename)
    except Exception as err:
        print(err)
        file_list.append(filepath)

    print(f'files len: {len(file_list)}')

    for i in range(len(file_list)):
        img = cv2.imread(f'{filepath}/{file_list[i]}', cv2.IMREAD_ANYCOLOR)
        img_width, img_height = img.shape[0], img.shape[1]
        print(f'img_width: {img_width}\t\timg_height: {img_height}')

        line_width = 0
        while True:
            img = cv2.line(img, (0, line_width), (img_height, line_width), (0, 255, 255), 2)
            line_width += width
            if line_width >= img_width:
                break

        line_height = 0
        while True:
            img = cv2.line(img, (line_height, 0), (line_height, img_width), (0, 255, 255), 2)
            line_height += height
            if line_height >= img_height:
                break

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        cv2.imwrite(f'{save_path}/ROI{width}x{height}_{file_list[i]}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    filepath = r'D:\Harry\000.DataAnalysis\002.python\ROI_IMAGE'
    save_folder = r'D:\Harry\000.DataAnalysis\003.ROI_image'
    make_line(source_path=filepath, save_path=save_folder, width=150, height=150)
