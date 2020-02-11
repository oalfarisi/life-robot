# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:54:28 2019
@author: Author2020Author
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
#import mglearn

#Clear the previous run in Console window
os.system('cls')
print("Start: os.system('cls')")



##############################
####Step1: Input ####
# Read CSV file
df=pd.read_csv('data\Data_Heterogeneous_20190206.csv',sep=',') #read the CSV file into df
df.columns=['PhiXSectContin','PixelColor','NeighbColorGrad','Betw2Amplify','Label','Label_text'] #Add header name to each column
print("shape=",df.shape)
print("df=\n",df)

from sklearn.model_selection import train_test_split
X_Train, X_Test,Y_Train,Y_Test=train_test_split(df.iloc[:,:4],df.iloc[:,4],random_state=0,test_size=0.1)

# create a scatter matrix from the dataframe, co by y_train
pd.plotting.scatter_matrix(X_Train, c=Y_Train, figsize=(8, 8),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_Train=", X_Train.shape, "Y_Train=", Y_Train.shape)
print("X_Test=", X_Test.shape, "Y_Test=", Y_Test.shape)
print("Keys of df:\n", df.keys())





##############################
####Step2: Use the methods ####

from sklearn.metrics import confusion_matrix
#Confusion matrix is [[00-TN,01-FP],[10-FN,11-TP]]
#Accuracy=(TN+TP)/(TN+FP+FN+TP); Precision=TP/(TP+FP);Recall=TP/(TP+FN)

class ConfMatrix(object):
    def __init__(self,the_conf_mat):
        self.Conf_matrix=the_conf_mat
        
        self.Accuracy=(self.Conf_matrix[0,0]+self.Conf_matrix[1,1])*1.0 / self.Conf_matrix.sum()
        self.Precision=(self.Conf_matrix[1,1])*1.0/ (self.Conf_matrix[1,1]+self.Conf_matrix[0,1])
        self.Recall=self.Conf_matrix[1,1]*1.0 / (self.Conf_matrix[1,1]+self.Conf_matrix[1,0])
        self.F1=2*self.Precision*self.Recall/(self.Precision+self.Recall)



print("\n\n###Step2.1 Try Knn method")
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=1)

Knn.fit(X_Train, Y_Train)

X_new = np.array([[1, 140, 80, 0]])
print("X_new.shape:", X_new.shape)

prediction = Knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       df['Label'][prediction])

Y_Pred_knn= Knn.predict(X_Test)
print("Test set predictions:\n", Y_Pred_knn)
print("Test set score: {:.2f}".format(np.mean(Y_Pred_knn == Y_Test)))
print("Test set score: {:.2f}".format(Knn.score(X_Test, Y_Test)))


#1) Checking Performance of user method ################
#Confusion_Omar= confusion_matrix(Y_Test,Test_df.loc[:,'Class_voted'])
Confusion_Knn= confusion_matrix(Y_Test,Y_Pred_knn)
print("Knn method Confusion Matrix:\n{}".format(Confusion_Knn))

###2)Calc.confusion_matrix for User method
ConfMatrix_Knn_obj=ConfMatrix(Confusion_Knn)
(Accuracy_Knn,Precision_Knn,Recall_Knn,F1_Knn)=(ConfMatrix_Knn_obj.Accuracy,ConfMatrix_Knn_obj.Precision,ConfMatrix_Knn_obj.Recall,ConfMatrix_Knn_obj.F1)
print("\nAccuracy_Knn=",Accuracy_Knn,",Precision_Knn=",Precision_Knn,",Recall_Knn=",Recall_Knn,",F1_Knn=",F1_Knn)



print("\n\n###Step2.2 Try SVM method")
from sklearn.svm import SVC
svc=SVC(C=100,gamma=0.1)
svc.fit(X_Train, Y_Train)
Y_Pred_svc=svc.predict(X_Test)

print("Test set predictions:\n", Y_Pred_svc)
print("Test set score: {:.2f}".format(np.mean(Y_Pred_svc == Y_Test)))
print("Test set score: {:.2f}".format(svc.score(X_Test, Y_Test)))

#1) Checking Performance of user method ################
#Confusion_Omar= confusion_matrix(Y_Test,Test_df.loc[:,'Class_voted'])
Confusion_svc= confusion_matrix(Y_Test,Y_Pred_svc)
print("SVC method Confusion Matrix:\n{}".format(Confusion_svc))

###2)Calc.confusion_matrix for User method
ConfMatrix_svc_obj=ConfMatrix(Confusion_svc)
(Accuracy_svc,Precision_svc,Recall_svc,F1_svc)=(ConfMatrix_svc_obj.Accuracy,ConfMatrix_svc_obj.Precision,ConfMatrix_svc_obj.Recall,ConfMatrix_svc_obj.F1)
print("\nAccuracy_svc=",Accuracy_svc,",Precision_svc=",Precision_svc,",Recall_svc=",Recall_svc,",F1_svc=",F1_svc)



print("\n\nThis is the end of code")



