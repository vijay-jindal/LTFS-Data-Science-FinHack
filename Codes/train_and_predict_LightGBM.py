# This code uses LightGBM algorithm for predicting loan defaults
# By Vijay Jindal : https://github.com/vijay-jindal/
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#MODEL TRAINING - LIGHTGBM - AUC = 0.6555

# Reading the training data and storing it as a dataframe
train=pd.read_csv('~/Processed_data/trainx.csv')
print("Training Data read")

# Reading the test data and storing it as a dataframe
test=pd.read_csv('~/Processed_data/testx.csv')
print("Testing Data read")

# Dropping the target and the ID of the customer to make final training dataset
X_train = train.drop(columns={'UniqueID','loan_default'},axis=1)

# Storing the target values in y_train for the data X_train
y_train = pd.DataFrame(train['loan_default'])

# Removing the ID column from the test dataset to test
test_x = test.drop(columns={'UniqueID'},axis=1)

# Splitting the data into training and validation data in ratio 4:1
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
print("Training data Split into training and validation.")

#Reshaping the labels data frame to match the model's requirements
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

print("Training Started")

# Importing the LightGBM library
import lightgbm as lgb

# Storing the categorical features (if any) as a list
categorical_features =[c for c, col in enumerate(train.columns) if 'cat' in col]
print(categorical_features)

# Creating the final training and testing datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
test_data = lgb.Dataset(X_test, label=y_test)

# Setting the hyper-parameters for the model
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth':50,
    'min_data_in_leaf':1000,
    'num_leaves':70,
    'max_bin':1100,
    'lambda':1,
    'is_unbalance': 'false',
    'boosting': 'gbdt',
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 30,
    'drop_rate': 0.2,
    'learning_rate': 0.005,
    'verbose': 0
}

# Training the model
model = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=8000,
                       early_stopping_rounds=100)

print("Training Completed\n")

# Predicting the results of the validation data using the model
y_pred = model.predict(X_test)

# Determining the ROC_AUC Score to know the model accuracy
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print("ROC_AUC SCORE : ",roc_auc)

# Predicting the results of the test data using the model
result=model.predict(test_x)

# Combining the predicted result to the Unique ID of the customer
prediction={'UniqueID': test['UniqueID'],'loan_default': result}

# Making a new dataframe with only the UniqueID of customer and the predicted result which are probabilities
df =pd.DataFrame(prediction, columns= ['UniqueID', 'loan_default'])

# Saving the final results as a csv in the file system.
file_name='light_prediction.csv'
df.to_csv(file_name,index=False)
