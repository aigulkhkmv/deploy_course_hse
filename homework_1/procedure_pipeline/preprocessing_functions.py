def text2num_sex(df):
    df["sex"] = df["sex"].replace("male", 0)
    df["sex"] = df["sex"].replace("female", 1)
    return df


def text2num_embarked(df):
    df["embarked"] = df["embarked"].replace("S", 0)
    df["embarked"] = df["embarked"].replace("C", 1)
    df["embarked"] = df["embarked"].replace("Q", 2)
    return df


def categorical2num(df):
    df = text2num_sex(df)
    df = text2num_embarked(df)
    return df


def df2array(df):
    X = df.to_numpy()
    return X
