import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 5627

df = pd.read_csv('data/2019_party_change_list.csv').drop_duplicates()
df.change_dt = pd.to_datetime(df.change_dt)

# Dropping rows with Libertarian, Green, or Constitutional party

def party_drop(party, dataframe):
    """This function removes unwanted political parties"""
    rows = dataframe.loc[(dataframe.party_from == party) | (dataframe.party_to == party)]
    dataframe = dataframe.drop(rows.index)
    return dataframe


dropped_parties = ['LIB', 'GRE', 'CST']
for party in dropped_parties:
    df = party_drop(party, df)


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

# Cross-Validate

# Print cleaned data
# cleaned_training_data = df.copy()
# cleaned_test_data = 
# pi_df.to_csv('df_cleaned.csv', sep=',')