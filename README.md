# LTFS-Data-Science-FinHack

## [About LTFS Data Science FinHack ( ML Hackathon)](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/)
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
