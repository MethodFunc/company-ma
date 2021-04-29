import pandas as pd
'''
기본 모듈 : pandas
 데이터 프레임의 정규화와 IQR을 계산하여 결측값을 제거하는 모듈
'''

def normalizer(dataframe, method='z_score'):
    '''
    :param dataframe: 기존 판다스 데이터 프레임
    :param method: min_max : min_max 정규화, z_score : z_score 정규화, 기본은 z_score입니다.
    :return: 정규화된 데이터 프레임
    '''
    # normalizer
    output = None

    if method == 'z_score':
        output = (dataframe - dataframe.mean(axis=1)) / dataframe.std(axis=1)

    if method == 'min_max':
        output = (dataframe - dataframe.min(axis=1)) / (dataframe.max(axis=1) - dataframe.min(axis=1))

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
