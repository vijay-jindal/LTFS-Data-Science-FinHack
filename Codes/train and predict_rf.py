# Importing the required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import resample

# Reading the training data and storing it as a dataframe
train=pd.read_csv('~/Processed_data/trainx.csv')
print("Training Data read")

# Reading the test data and storing it as a dataframe
test=pd.read_csv('~/Processed_data/testx.csv')
print("Testing Data read")

# Storing the part of data with loan_default=0 in df_majority
df_majority = train[train.loan_default==0]

# Storing the part of data with loan_default=0 in df_minority
df_minority = train[train.loan_default==1]

# Resampling the data to increase the number of reading with loan_default = 1
df_up = resample(df_minority,replace=True,n_samples=182543,random_state=123)

# Combining the sampled data and df_majority
train = pd.concat([df_majority,df_up])

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

# Here we are using RandomForest Classifier
model=RandomForestClassifier(n_estimators=300,max_features=0.9,max_depth=100,min_samples_split=3,min_samples_leaf=50, n_jobs=-1)

# Fitting the model to the train dataset
model.fit(X_train,y_train)

print("Training Completed\n")

# Saving the model in the file system to use later by using the pickle library
pkl_filename = "rf_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print("Model saved to disk with name xgb_model.pkl\n")

# Predicting the results from the validation data
y_pred = model.predict(X_test)

# Determining the ROC_AUC Score to know the model accuracy
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print("ROC_AUC SCORE : ",roc_auc)

# Predicting the results of the test data using the model
result=model.predict(test_x)

# Combining the predicted result to the Unique ID of the customer
prediction={'UniqueID': test['UniqueID'],'loan_default': result}

# Making a new dataframe with only the UniqueID of customer and the predicted result which is 0 or 1.
df =pd.DataFrame(prediction, columns= ['UniqueID', 'loan_default'])

# Saving the final results as a csv in the file system.
df.to_csv('rf_prediction.csv',index=False)

print("Completed. File saved.\n")

# Determining the AUC with predict_proba to know model accuracy
probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)

print('AUC with proba : %.3f' % auc)

# Predicting the results of the test data using the model as probabilities
result=model.predict_proba(test_x)
result = result[:, 1]

# Combining the predicted result to the Unique ID of the customer
prediction={'UniqueID': test['UniqueID'],'loan_default': result}

# Making a new dataframe with only the UniqueID of customer and the predicted result which are probabilities
df =pd.DataFrame(prediction, columns= ['UniqueID', 'loan_default'])

# Saving the final results as a csv in the file system.
df.to_csv('prediction.csv',index=False)

print("Completed. File saved. prediction.csv\n")
