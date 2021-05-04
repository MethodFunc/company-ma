import os
import cv2
import glob
import mahotas as mt
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern
from scipy import stats

'''
기본 모듈 : pandas
 데이터 프레임의 정규화와 IQR을 계산하여 결측값을 제거하는 모듈
'''


def ttest_2samp(df1, df2):
    p_value_05 = {}

    for x in df1.columns:
        t_value = stats.ttest_ind(df1[x], df2[x]).statistic
        p_value = stats.ttest_ind(df1[x], df2[x]).pvalue

        print(f'{x} = t_value: {t_value:.4f}, p_value: {p_value:.4f}')

        if p_value < 0.05:
            p_value_05[x] = p_value

    print(f'0.05이하 채택 된 {p_value_05.keys()}')


def normalizer(dataframe, method='z_score'):
    '''
    :param dataframe: 기존 판다스 데이터 프레임
    :param method: min_max : min_max 정규화, z_score : z_score 정규화, 기본은 z_score입니다.
    :return: 정규화된 데이터 프레임
    '''
    # normalizer

    if method == 'z_score':
        output = (dataframe - dataframe.mean()) / dataframe.std()

    if method == 'min_max':
        output = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

    return output


def calc_IQR(dataframe, fill_miss=None):
    '''
    :param dataframe: 기존 판다스 데이터 프레임
    :param fill_miss: mean : 결측값을 평균으로 채움, zero : 결측값을 0으로 채움, drop : 결측값 행을 모두 제거함, 기본값은 None이다.
    :return: fill_miss에 따른 데이터프레임 반환
    '''
    df_dec = dataframe.describe().T
    df_dec['IQR'] = df_dec['75%'] - df_dec['25%']
    df_dec['IQR_min'] = df_dec['25%'] - 1.5 * df_dec['IQR']
    df_dec['IQR_max'] = df_dec['75%'] + 1.5 * df_dec['IQR']

    rd = dataframe[(dataframe < df_dec['IQR_max'])]
    output = rd[(dataframe > df_dec['IQR_min'])]

    if fill_miss == 'mean':
        output = rd.fillna(rd.mean())

    if fill_miss == 'zero':
        output = rd.fillna(0)

    if fill_miss == 'drop':
        output = rd.dropna(axis=0)

    return output


def calc_mann(df_name, df, df2):
    ''' 2 dataframe 비모수 검정(lbp, haralick)'''
    globals()[f"{df_name}_pvalue"] = {}

    for col in df.columns:
        s, p = stats.mannwhitneyu(df[col], df2[col])
        if p >= 0.05:
            print(f"{col}의 p_value값은 {p:.4f}으로 귀무가설이 기각되지 않았으므로 편광필터에 의한 차이가 없다고 할 수 있다.")

        if p < 0.05:
            globals()[f"{df_name}_pvalue"][col] = f"{p:.4f}"


def feature_extract_hl(source_path, categories, roi, image_sample):
    if "\\" in source_path:
        source_path = source_path.replace("\\", "/")
    eps = 1e-7
    width, height = 150, 150
    hal_col = ['hal1', 'hal2', 'hal3', 'hal4', 'hal5', 'hal6', 'hal7', 'hal8', 'hal9', 'hal10', 'hal11', 'hal12',
               'hal13']
    lbp_col = ['lbp1', 'lbp2', 'lbp3', 'lbp4', 'lbp5', 'lbp6', 'lbp7', 'lbp8', 'lbp9', 'lbp10']
    for cat in categories:
        path = f'{source_path}/{cat}'
        if not os.path.isdir(path):
            print(f"Folder {cat} is not exist")
            print(f"Please check your categories")
            break

        file_list = glob.glob(f"{path}/*.jpg")

        if len(file_list) < image_sample or len(file_list) == 0:
            print(f"{cat} 이미지 파일 갯수가 {image_sample}보다 모자름")
            break

        lbp_list, hal_list = [], []
        for imgs in file_list[:image_sample]:
            img = cv2.imread(imgs)
            for (i, j) in roi:
                x, y = i * width, j * height
                roi_img = img[y:y + height, x:x + width]
                roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                haralick = mt.features.haralick(roi_gray).mean(axis=0)
                lbp = local_binary_pattern(roi_gray, 8, 3, method='uniform')
                lbp = np.array(lbp)
                (hist, _) = np.histogram(lbp.ravel(), density=True, bins=int(lbp.max() + 1),
                                         range=(0, int(lbp.max() + 1)))
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)

                lbp_list.append(hist)
                hal_list.append(haralick)

        globals()[f'{cat}_lbp'] = lbp_list
        globals()[f'{cat}_lbp_pd'] = pd.DataFrame(lbp_list, columns=lbp_col)

        globals()[f'{cat}_hal'] = hal_list
        globals()[f'{cat}_hal_pd'] = pd.DataFrame(hal_list, columns=hal_col)
