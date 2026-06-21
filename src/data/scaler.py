from sklearn.preprocessing import StandardScaler

def scale_data(
        train_X,
        test_X
):

    scaler = StandardScaler()
    scaler.fit(train_X)

    return(scaler.transform(train_X),
           scaler.transform(test_X),
           scaler
    )
