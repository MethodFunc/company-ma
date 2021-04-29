
def Outlier(df):
    df_desc = df.describe().T
    df_desc['IQR'] = df_desc['75%'] - df_desc['25%']
    df_desc['IQR_min'] = df_desc['25%'] - 1.5 * df_desc['IQR']
    df_desc['IQR_max'] = df_desc['75%'] + 1.5 * df_desc['IQR']

    true_df = df[df < df_desc['IQR_max']]
    true_df = true_df[df > df_desc['IQR_min']]

    output = true_df.fillna(true_df.mean())

    return output