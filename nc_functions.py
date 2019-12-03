"""This file holds all of our functions"""


def party_drop(party, dataframe):
    """This function removes unwanted political parties"""
    rows = dataframe.loc[(dataframe.party_from == party) | (dataframe.party_to == party)]
    dataframe = dataframe.drop(rows.index)
    return dataframe
