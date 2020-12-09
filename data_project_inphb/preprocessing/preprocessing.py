def convert_cast(df, categorical_cols):
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype("object")
    return df

def convert_cast(mapper_config=None, dataframe=None):
    """
    Convert and cast DataFrame field types.

    This function achieves the same purpose as pd.read_csv, but it is used on
    actual DataFrame obtained from SQL.
    """
    # field conversion
    for var_name, lambda_string in mapper_config['converters'].items():
        if var_name in dataframe.columns:
            func = eval(lambda_string)
            dataframe[var_name] = dataframe[var_name].apply(func)

    # field type conversion/compression
    dty = dict(dataframe.dtypes)
    for col_name, actual_type in dty.items():
        expected_type = mapper_config['columns'][col_name]['dtype']
        if actual_type != expected_type:
            dataframe[col_name] = dataframe[col_name].astype(expected_type)

    return dataframe

def input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype is float) or (df[col].dtype is int):
            df[col] = df[col].fillna(df[col].median())
        if (df[col].dtype == object):
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def parse_model(X, use_columns):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X = X[use_columns]
    return X, target


def transform_df(X, columns_to_dummify, features=["Pclass"], thres=10):
    X = convert_df_columns( X, features, type_var="object")
    X["is_child"] = X["Age"].apply(lambda x: 0 if x < thres else 1)
    X["title"] = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    X['surname'] = X['Name'].map(lambda x: '(' in x)
    for col in columns_to_dummify:
        X_dummies = pd.get_dummies(X[col], prefix=col,
                                   drop_first=False, dummy_na=False, prefix_sep='_')
        X = X.join(X_dummies).drop(col, axis=1)
    return X.drop("Name", axis=1).drop("Age", axis=1)   

def dummify_features(df):
    """
    Transform categorical variables to dummy variables.

    Parameters
    ----------
    df: dataframe containing only categorical features

    Returns
    -------
    X: new dataframe with dummified features
       Each column name becomes the previous one + the modality of the feature

    enc: the OneHotEncoder that produced X (it's used later in the processing chain)
    """
    
    from sklearn import preprocessing

    colnames = df.columns
    le_dict = {}
    for col in colnames:
        le_dict[col] = preprocessing.LabelEncoder()
        le_dict[col].fit(df[col])
        df.loc[:, col] = le_dict[col].transform(df[col])

    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    X = enc.transform(df)

    dummy_colnames = [cv + '_' + str(modality) for cv in colnames for modality in le_dict[cv].classes_]
    # for cv in colnames:
    #     for modality in le_dict[cv].classes_:
    #         dummy_colnames.append(cv + '_' + modality)

    return X, dummy_colnames, enc