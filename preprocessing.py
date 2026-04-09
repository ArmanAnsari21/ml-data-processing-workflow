def preprocess_data(df):
    df = df.copy()

    df = df.dropna()

    df["feature_sum"] = df["feature1"] + df["feature2"]

    return df
