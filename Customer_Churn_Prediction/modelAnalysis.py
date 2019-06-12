#Imports
import os.path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


#Load dataset('Churn-train' in this case)
dataset = pd.read_csv(os.path.join('Datasets', 'churn-train.csv'))
#print(dataset.head(9))


#Divide data into class and numerical values
x = dataset.iloc[:,[0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
y = dataset.iloc[:,-1]


#Split data into test and train data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)


#Scaling to convert 1 to CHurn and 0 to No Churn
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


#Classify the dataset using various models
#RandomForestClassifier
rfc_classifier = RandomForestClassifier(n_estimators=70, random_state=1, max_depth=16)   
rfc_classifier.fit(X_train, y_train)
 
#knn classifier with k=17
knn_classifier = KNeighborsClassifier(n_neighbors=17, n_jobs=3)
knn_classifier.fit(X_train, y_train) 

#glm model
glm_classifier = LogisticRegression()
glm_classifier.fit(X_train, y_train)

#svm model
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)


#predict probability and classes using all the above used models
#RandomForestClassifier
rfc_class = rfc_classifier.predict(X_test)
rfc_prob = rfc_classifier.predict_proba(X_test)

#knn
knn_prob = knn_classifier.predict_proba(X_test)
knn_class = knn_classifier.predict(X_test)

#glm
glm_prob = glm_classifier.predict_proba(X_test)
glm_class = glm_classifier.predict(X_test)

#svm
svm_prob = svm_classifier.predict_proba(X_test)
svm_class = svm_classifier.predict(X_test)


#calculate metrics for for each model to defiine accuracy
#rfc
print("RFC:")
print ('Confusion Matrix :')
print(confusion_matrix(y_test, rfc_class) )
print ('Report : ')
print (classification_report(y_test, rfc_class))
print ('Accuracy Score :',accuracy_score(y_test, rfc_class)) 
print ('AUC Score :',roc_auc_score(y_test,rfc_prob[:,1])) 

#knn
print("KNN:")
print ('Confusion Matrix :')
print(confusion_matrix(y_test, knn_class) )
print ('Report : ')
print (classification_report(y_test, knn_class)) 
print ('Accuracy Score :',accuracy_score(y_test, knn_class)) 
print ('AUC Score :',roc_auc_score(y_test,knn_prob[:,1])) 

#glm
print("GLM:")
print ('Confusion Matrix :')
print(confusion_matrix(y_test, glm_class) )
print ('Report : ')
print (classification_report(y_test, glm_class))
print ('Accuracy Score :',accuracy_score(y_test, glm_class)) 
print ('AUC Score :',roc_auc_score(y_test,glm_prob[:,1])) 

#svm
print("SVM:")
print ('Confusion Matrix :')
print(confusion_matrix(y_test, svm_class) )
print ('Report : ')
print (classification_report(y_test, svm_class))
print ('Accuracy Score :',accuracy_score(y_test, svm_class)) 
print ('AUC Score :',roc_auc_score(y_test,svm_prob[:,1])) 


#plot the roc curve
#Obtain true positive rate (tpr), false positive rate (fpr), and the corresponding threshold
rfc_roc = roc_curve(y_true=y_test, y_score=rfc_prob[:,1])
knn_roc = roc_curve(y_true=y_test, y_score=knn_prob[:,1])
glm_roc = roc_curve(y_true=y_test, y_score=glm_prob[:,1])
svm_roc = roc_curve(y_true=y_test, y_score=svm_prob[:,1])

plt.plot(rfc_roc[0], rfc_roc[1])
plt.plot(knn_roc[0], knn_roc[1])
plt.plot(glm_roc[0], glm_roc[1])
plt.plot(svm_roc[0], svm_roc[1])


# Label the x- and y-axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# Create a legend for the all curves
legend_text = ['RFC (AUC = {:.2f}%)'.format(roc_auc_score(y_test,rfc_prob[:,1])*100),
'KNN (AUC = {:.2f}%)'.format(roc_auc_score(y_test,knn_prob[:,1])*100),
'GLM (AUC = {:.2f}%)'.format(roc_auc_score(y_test,glm_prob[:,1])*100),
'SVM (AUC = {:.2f}%)'.format(roc_auc_score(y_test,svm_prob[:,1])*100)]

plt.legend(legend_text, title='Model')

# Set a sensible plot title
plt.title('Churn dataset')

#Turn on the grid
plt.grid()
#plt.show()

#Save the plot as a PNG file in the `results_dir'
img = os.path.join('Results', 'TestResults', 'comparisonGraph.png')
plt.savefig(img)
