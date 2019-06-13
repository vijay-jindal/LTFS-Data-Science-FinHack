from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv("Input_data/train.csv")

# Replacing null values in the employment with salaried
data.fillna('Salaried', inplace=True)
# Creating a new feature named 'age' by subtracting the loan year from the date of birth
data['age']=None
data['age'] = data['Date.of.Birth'].map(lambda x:int(x[-2:])-18)
data.drop(columns={"Date.of.Birth",'DisbursalDate'},inplace=True)

# Converting all the data to integers
data['disbursed_amount']= [int(m) for m in data['disbursed_amount']]
data['asset_cost']= [int(m) for m in data['asset_cost']]

# Converting the data to float
data['ltv']= [float(m) for m in data['ltv']]
data['PRI.CURRENT.BALANCE']= [float(m) for m in data['PRI.CURRENT.BALANCE']]
data['PRI.SANCTIONED.AMOUNT']= [float(m) for m in data['PRI.SANCTIONED.AMOUNT']]

# Using one hot encode to represent the CNS score and employment type into certain possible classes
def encode(datax, labels):
    for label in labels:
        datax = datax.join((pd.get_dummies(datax[label], prefix = label)))
        datax.drop(label, axis=1, inplace=True)
    return datax

data = encode(data, ['Employment.Type'])
data = data.drop(['PERFORM_CNS.SCORE.DESCRIPTION'],axis=1)

data["AVERAGE.ACCT.AGE"] = data["AVERAGE.ACCT.AGE"]\
.map(lambda x:(int(x.split()[0][0])*12.0)+(int(x.split()[1][0])))

data["CREDIT.HISTORY.LENGTH"] = data["CREDIT.HISTORY.LENGTH"]\
.map(lambda x:(int(x.split()[0][0])*12.0)+(int(x.split()[1][0])))

# Storing the final result as a csv file
data.to_csv('trainx.csv')


