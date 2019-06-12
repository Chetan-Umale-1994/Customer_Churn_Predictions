#Using Random Forest model to predict the "Churn-holdout" data.

#Import statements
import os.path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

#Load train and holdout(test) data
#data_dir = os.path.join('Data')
dataset_train = pd.read_csv(os.path.join('Datasets', 'churn-train.csv'))
dataset_test = pd.read_csv(os.path.join('Datasets', 'churn-holdout.csv'))

#Separate class variable from other numerical data
X_train = dataset_train.iloc[:,[0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
X_test = dataset_test.iloc[:,[0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
Y_train = dataset_train.iloc[:,-1]

#Define the identifier(phone number in this case)
identifier = dataset_train.iloc[:,3]

#Classify the test data using random forest classifier using 90 trees and 1 as the seed value.
rf_classifier = RandomForestClassifier(n_estimators=70, random_state=1, max_depth=16)   
rf_classifier.fit(X_train, Y_train) 

#Predict the classified data as Churn or NoChurn. 
Y_pred = rf_classifier.predict(X_test)

#Calculate the probability of Churn and Nochurn
churn_prob = rf_classifier.predict_proba(X_test)

#Create data frame to save the results
prob_rfc = pd.DataFrame(churn_prob, columns=rf_classifier.classes_)
prob_rfc['Predictions'] = Y_pred
prob_rfc['Phone Number'] = identifier

#Rearrange the order of column in the result file.
prob_rfc = prob_rfc[['Phone Number', 'Predictions', 0, 1]]

#Generate csv file for results
prob_rfc.to_csv(os.path.join('Results','FinalPredictions', '118221161_118221388_118220570.csv'), index=False)
