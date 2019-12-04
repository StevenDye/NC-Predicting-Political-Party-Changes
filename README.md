# FIS-Mod4-Project: NC-Voting-Data

This project focuses on logistic regression testing using 2019 voter registration data from the state of North Carolina

Date: December 2019

Project Members: Irv Campbell and Steven Dye

Goal: To be able to predict what political party a voter will change their registration to based on what party they were previously registered to and what county they live in.

Responsibilities:
 - Define project scope and focus
 - Collect data
 - Form hypothesis
 - Perform exploratory data analysis
 - Create Master Notebook
 - Create Regression Model
 - Test hypothesis
 - Create presentation
 - Lint/clean code file
 
 Summary of files:
 - Master_Notebook.ipynb: Jupyter Notebook documenting the code and the analysis for the project. Written for a technical audience
 - Predicting N.C. Voter Party Changes.pdf: PDF of final presentation
 - data file
     - 2019_party_change_list.csv: 2019 Voter registration data from the state 
     - X_test.csv: Test features
     - X_train.csv: Train features with SMOTE
     - y_test.csv: Test target
     - y_train.csv: Train target with SMOTE
- data_prep.py: Code used to clean data and to add SMOTE data
- nc_functions.py: Module to store functions
- viz.py: File for storing vizualization functions
