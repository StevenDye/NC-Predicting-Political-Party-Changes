"""This file cleans data/2019_party_change_list.cs and outouts X_train.csv,
X_test.csv, y_train.csv, and y_test.csv"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from nc_functions import party_drop

RANDOM_STATE = 5627

df = pd.read_csv('data/2019_party_change_list.csv').drop_duplicates()
df.change_dt = pd.to_datetime(df.change_dt)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

# SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)

# Convert to dataframes
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)

# Print cleaned data
X_train_df.to_csv('data/X_train.csv', sep=',')
y_train_df.to_csv('data/y_train.csv', sep=',')
X_test_df.to_csv('data/X_test.csv', sep=',')
y_test_df.to_csv('data/y_test.csv', sep=',')
