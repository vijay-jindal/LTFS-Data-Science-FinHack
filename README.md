![alt text](https://github.com/vijay-jindal/LTFS-Data-Science-FinHack/blob/master/ml_hack_cover_image.jpg)

# [LTFS-Data-Science-FinHack](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/)

## About LTFS Data Science FinHack ( ML Hackathon)
L&T Financial Services & Analytics Vidhya presents ‘DataScience FinHack’.

In this FinHack, you will develop a model for our most common but real challenge ‘Loan Default Prediction’ & also, get a feel of our business!
If your solution adds good value to our organization, take it from us, Sky is the limit for you!

## Problem Statement

Financial institutions face large amount of losses due to vehicle loan defaults. The problem statement is to predict the loan default based on the information provided by the Loanee. The dataset contains the following data :
Loan Details Such as the loan amount, loan/value, date of disbursal and other related details.
Information of the Loanee such as the ID, Date of birth, Location, Number of accounts and other related details.

By predicting the candidates who might get involved with a loan default, the financial institutions can take steps to reduce the default by cancelling the loan or by any other means.

## About the Data
- **Train.csv** file contains all data about the Loanee and the loan along with the loan_default.
- **Test.csv** file contains all data about the loanee and the loan, but does not contain the loan_default since it has to be predicted.
- **Sample_submission.csv** contains a sample of how the final submission should look.

## Final Output
The final output of the code should be a .csv file with the Customer ID and the loan_default value(either in 0/1 or a **probability**)

## Evaluation Metric
The submissions are evaluated based on the **area under the ROC curve** between the predicted value and the actual value.

## Approach
The project uses 3 different algorithms to predict loan defaults. The one with the highest area under the roc curve value is chosen and the submission is made. The Algorithms used are :
- Random Forest
- LightGBM
- Adaboost
To deploy the codes follow the below steps :
1. Download the repository
2. Run the preprocessing code both on training and testing dataset as : python Pre_processing.py
3. Run the python files for training the model and predicting the loan defaults as : python train_and_predict_Adaboost.py
4. Done.

## Leaderboard

### [Private Leaderboard](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/pvt_lb) : 129/1339
### [Public Leaderboard](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/lb) : 151/1339
