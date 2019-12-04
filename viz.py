"""
This file holds the visualization functions for the NC voter
change data.

Extended description of function.
confusion_matrix_heat_map creates a heat map representing
a normalized confusion matrix.
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def make_percent_df(df):
    dict_of_party_values = df.party_to.value_counts().to_dict()
    raw_data_normal = {k: round(v/(sum(dict_of_party_values.values())), 2) for k, v in dict_of_party_values.items()}
    df_percent = pd.DataFrame(list(raw_data_normal.items()))
    df_percent.rename(columns={0:'Party', 1:'Percent of Voters'}, inplace = True)
    df_percent = df_percent.style.hide_index()
    return(df_percent)


def confusion_matrix_heat_map(clf, X_test, y_test):
    """This function takes a classifier clf as an argument to create a
    confusion matrix.  The function then normalizes the data across
    the confusion matrix and corrects a known bug in matplotlib that
    incorrectly cuts off the top and bottom rows of the heat map.

    Parameters:
    clf: Classifier already fit to training data
    X_test: Features for test data
    y_test: Predicted outputs for X_test

    """
    # create confusion matrix <cm>
    cm = metrics.confusion_matrix(clf.predict(X_test), y_test)
    # create normalized confusion matrix <cm_nor>
    cm_nor = np.zeros((cm.shape[0], cm.shape[1]))
    for col in range(cm.shape[1]):
        cm_nor[:, col] = (cm[:, col] / sum(cm[:, col]))
    plt.ylim(-10, 10)
    # create normalized confusion matrix heat map
    sns.heatmap(cm_nor, cmap="Blues", annot=True, annot_kws={"size": 8})
    locs, labels = plt.xticks()
    plt.xticks(locs, ("DEM", "REP", "UNA"))
    locs, labels = plt.yticks()
    plt.yticks(locs, ("DEM", "REP", "UNA"))
    plt.yticks(rotation=0)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Voter Party Change")
    # known bug in matplotlib chops off a portion of the
    # top and bottom rows of heat maps.  This section of
    # code recovers the top and bottom limits and moves them
    # so that the map displays appropriately.
    bottom, top = plt.ylim()
    bottom += 0.5
    top -= 0.5
    plt.ylim(bottom, top)
    plt.show()
