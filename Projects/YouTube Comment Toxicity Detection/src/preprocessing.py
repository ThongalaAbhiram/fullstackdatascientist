from sklearn.preprocessing import LabelEncoder

def preprocess(df):

    le_region = LabelEncoder()
    le_category = LabelEncoder()
    le_product_name = LabelEncoder()
    le_month = LabelEncoder()
    le_holiday = LabelEncoder()

    # Encode categorical columns

    df['Region'] = le_region.fit_transform(
        df['Region']
    )

    df['Product_Category'] = le_category.fit_transform(
        df['Product_Category']
    )

    df['Product_Name'] = le_product_name.fit_transform(
        df['Product_Name']
    )

    df['Month'] = le_month.fit_transform(
        df['Month']
    )

    df['Holiday_Season'] = le_holiday.fit_transform(
        df['Holiday_Season']
    )

    return df