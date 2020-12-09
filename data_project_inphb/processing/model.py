def compute_score(clf, X, y, cv=5):
    """compute score in a classification modelisation.
       clf: classifier
       X: features
       y: target
    """
    xval = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (xval.mean(), xval.std() * 2))
    return xval


def plot_hist(feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=["Victime", "Survivant"], bins=bins, color=['r', 'b'])
    plt.legend(loc="upper left")
    plt.title('distribution relative de %s' %feature)
    plt.show()
    
def My_model ( X, y, size, RdomState = 42) :
    #X, y
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=size, 
                                                        random_state=RdomState )
    model = LogisticRegression(random_state= RdomState, max_iter=1000)
    model.fit(X_train, y_train)
    # Run the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    metric = metrics.classification_report(y_test, ypred)

    return {"y_test": y_test, "prediction": y_pred, "proba":y_prob,
            "score_train": score_train, "score_test": score_test,
            "model": model, "metric": print(metric)}
    
    
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

def parse_model2(X):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    to_dummy = ['Pclass', 'Sex']
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix=dum)
        for col in split_temp:
            X[col] = split_temp[col]
        del X[dum]
    X['Age'] = X['Age'].fillna(X['Age'].median())
    to_del = ["PassengerId", "Name", "Cabin", "Embarked", "Survived", "Ticket"]
    for col in to_del:
        del X[col]
    return X, target