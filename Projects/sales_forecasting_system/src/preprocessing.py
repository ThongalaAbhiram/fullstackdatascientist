from sklearn.preprocessing import LabelEncoder

def preprocess(df):

    le_region = LabelEncoder()
    le_product = LabelEncoder()
    le_month = LabelEncoder()

    df['Region'] = le_region.fit_transform(df['Region'])
    df['Product'] = le_product.fit_transform(df['Product'])
    df['Month'] = le_month.fit_transform(df['Month'])

    return df