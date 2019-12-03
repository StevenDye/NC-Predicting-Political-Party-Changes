"""This file cleans data/2019_party_change_list.csv and returns X_train,
X_test, y_train, and y_test"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from nc_functions import party_drop


def data_preprocessing():
    """This file cleans data/2019_party_change_list.csv
    and returns X_train, X_test, y_train, and y_test"""
    rand_state = 5627

    df = pd.read_csv('data/2019_party_change_list.csv').drop_duplicates()

    # Dropping rows with Libertarian, Green, or Constitutional party
    dropped_parties = ['LIB', 'GRE', 'CST']
    for pol_party in dropped_parties:
        df = party_drop(pol_party, df)

    # Hot encode
    X, y = df[['county_id', 'party_from']], df.party_to
    ncencoder = preprocessing.OneHotEncoder()
    nclabels = preprocessing.LabelEncoder()
    X = ncencoder.fit_transform(X)
    nclabels.fit(y)
    y = nclabels.transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand_state)

    # SMOTE
    smt = SMOTE()
    X_train, y_train = smt.fit_sample(X_train, y_train)

    return X_train, y_train, X_test, y_test
