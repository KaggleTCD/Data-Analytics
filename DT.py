# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:15:24 2019

@author: SYSTEM
"""

#import modules
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


#Read the excel dataset
#"F:/MS/Masters/Machine Learning/"
df = pd.read_excel("Project Data.xlsx")

#Replacing the null values of X by median
df["X1"].fillna(df.X1.median(),inplace = True)
df["X2"].fillna(df.X2.median(),inplace = True)
df["X3"].fillna(df.X3.median(),inplace = True)
df["X5"].fillna(df.X5.median(),inplace = True)
df["X6"].fillna(df.X6.median(),inplace = True)
df["X7"].fillna(df.X7.median(),inplace = True)


#Replacing the null values of Y by mode
df["Y1"].fillna(df.Y1.mode()[0],inplace = True)
df["Y2"].fillna(df.Y2.mode()[0],inplace = True)
df["Y3"].fillna(df.Y3.mode()[0],inplace = True)
df["Y5"].fillna(df.Y5.mode()[0],inplace = True)
df["Y6"].fillna(df.Y6.mode()[0],inplace = True)
df["Y7"].fillna(df.Y7.mode()[0],inplace = True)


#Dropping ID column as it is irrelevant
df = df.drop("ID",axis = 1)

df_response = df.iloc[:,0:1]

#Make a DecisionTreeClassifier object
dtc = DecisionTreeClassifier(max_depth = 4)
#Visualization
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
def fitAndVisualize(df_variables,df_response,treeName):
    model = dtc.fit(df_variables,df_response)
    dot_data = StringIO()       
    export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = df_variables.columns,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(treeName)
    Image(graph.create_png())
    return model
    
#Including only 0,1 and X as feature variables
feature_variables = df.drop("Response",axis=1)
feature_variables = feature_variables.iloc[:,0:8]
X01X_train,X01X_test,Y01X_train,Y01X_test = train_test_split(feature_variables,df_response,test_size=0.2)
model = fitAndVisualize(X01X_train,Y01X_train,"0 1and X.png")
Y_pred = model.predict(X01X_test)
print("01X confusion matrix ",confusion_matrix(Y01X_test,Y_pred))
print()
print("01X accuracy matrix ",accuracy_score(Y01X_test,Y_pred))
print()

#Including only 0,1 and Y as feature variables
feature_variables = df.drop(["Response","X1","X2","X3","X4","X5","X6","X7"],axis=1)
X01Y_train,X01Y_test,Y01Y_train,Y01Y_test = train_test_split(feature_variables,df_response,test_size=0.2)
model = fitAndVisualize(X01Y_train,Y01Y_train,"0 1and Y.png")
Y_pred = model.predict(X01Y_test)
print("01Y confusion matrix ",confusion_matrix(Y01Y_test,Y_pred))
print()
print("01Y accuracy matrix ",accuracy_score(Y01Y_test,Y_pred))
print()

#Including only 0,1 and X and Y as feature variables
feature_variables = df.drop("Response",axis=1)
X01XY_train,X01XY_test,Y01XY_train,Y01XY_test = train_test_split(feature_variables,df_response,test_size=0.2)
model = fitAndVisualize(X01XY_train,Y01XY_train,"0 1and XY.png")
Y_pred = model.predict(X01XY_test)
print("01XY confusion matrix ",confusion_matrix(Y01XY_test,Y_pred))
print()
print("01XY accuracy matrix ",accuracy_score(Y01XY_test,Y_pred))
print()

#Including only 0 and X as feature variables
feature_variables_group0 = df.drop(["Y1","Y2","Y3","Y4","Y5","Y6","Y7"],axis=1)
feature_variables_group0 = feature_variables_group0[feature_variables_group0.Group==0]
print(feature_variables_group0)
df_response = feature_variables_group0.iloc[:,0:1]
print(df_response)
feature_variables_group0 = feature_variables_group0.drop(["Response"],axis=1)
X0X_train,X0X_test,Y0X_train,Y0X_test = train_test_split(feature_variables_group0,df_response,test_size=0.2)
model = fitAndVisualize(X0X_train,Y0X_train,"0 and X.png")
Y_pred = model.predict(X0X_test)
print("0X confusion matrix ",confusion_matrix(Y0X_test,Y_pred))
print()
print("0X accuracy matrix ",accuracy_score(Y0X_test,Y_pred))
print(feature_variables_group0.columns)



#Including only 0 and Y as feature variables
feature_variables_group0 = df.drop(["X1","X2","X3","X4","X5","X6","X7"],axis=1)
feature_variables_group0 = feature_variables_group0[feature_variables_group0.Group==0]
print(feature_variables_group0)
df_response = feature_variables_group0.iloc[:,0:1]
print(df_response)
feature_variables_group0 = feature_variables_group0.drop(["Response"],axis=1)
X0Y_train,X0Y_test,Y0Y_train,Y0Y_test = train_test_split(feature_variables_group0,df_response,test_size=0.2)
model = fitAndVisualize(X0Y_train,Y0Y_train,"0 and Y.png")
Y_pred = model.predict(X0Y_test)
print("0Y confusion matrix ",confusion_matrix(Y0Y_test,Y_pred))
print()
print("0Y accuracy matrix ",accuracy_score(Y0Y_test,Y_pred))
print(feature_variables_group0.columns)

#Including only 0,X and Y as feature variables
feature_variables_group0 = df[df.Group==0]
print(feature_variables_group0)
df_response = feature_variables_group0.iloc[:,0:1]
print(df_response)
feature_variables_group0 = feature_variables_group0.drop("Response",axis=1)
X0XY_train,X0XY_test,Y0XY_train,Y0XY_test = train_test_split(feature_variables_group0,df_response,test_size=0.2)
model = fitAndVisualize(X0XY_train,Y0XY_train,"0 X and Y.png")
Y_pred = model.predict(X0XY_test)
print("0XY confusion matrix ",confusion_matrix(Y0XY_test,Y_pred))
print()
print("0XY accuracy matrix ",accuracy_score(Y0XY_test,Y_pred))
print(feature_variables_group0.columns)



#Including only 1 and X as feature variables
feature_variables_group1 = df.drop(["Y1","Y2","Y3","Y4","Y5","Y6","Y7"],axis=1)
feature_variables_group1 = feature_variables_group1[feature_variables_group1.Group==1]
print(feature_variables_group1)
df_response = feature_variables_group1.iloc[:,0:1]
print(df_response)
feature_variables_group1 = feature_variables_group1.drop(["Response"],axis=1)
X1X_train,X1X_test,Y1X_train,Y1X_test = train_test_split(feature_variables_group1,df_response,test_size=0.2)
model = fitAndVisualize(X1X_train,Y1X_train,"1 and X.png")
Y_pred = model.predict(X1X_test)
print("1X confusion matrix ",confusion_matrix(Y1X_test,Y_pred))
print()
print("1X accuracy matrix ",accuracy_score(Y1X_test,Y_pred))
print(feature_variables_group1.columns)



#Including only 1 and Y as feature variables
feature_variables_group1 = df.drop(["X1","X2","X3","X4","X5","X6","X7"],axis=1)
feature_variables_group1 = feature_variables_group1[feature_variables_group1.Group==1]
print(feature_variables_group1)
df_response = feature_variables_group1.iloc[:,0:1]
print(df_response)
feature_variables_group1 = feature_variables_group1.drop(["Response"],axis=1)
X1Y_train,X1Y_test,Y1Y_train,Y1Y_test = train_test_split(feature_variables_group1,df_response,test_size=0.2)
model = fitAndVisualize(X1Y_train,Y1Y_train,"1 and Y.png")
Y_pred = model.predict(X1Y_test)
print("1Y confusion matrix ",confusion_matrix(Y1Y_test,Y_pred))
print()
print("1Y accuracy matrix ",accuracy_score(Y1Y_test,Y_pred))
print(feature_variables_group1.columns)


#Including only 1,X and Y as feature variables
feature_variables_group1 = df[df.Group==1]
print(feature_variables_group1)
df_response = feature_variables_group1.iloc[:,0:1]
print(df_response)
feature_variables_group1 = feature_variables_group1.drop("Response",axis=1)
X1XY_train,X1XY_test,Y1XY_train,Y1XY_test = train_test_split(feature_variables_group1,df_response,test_size=0.2)
model = fitAndVisualize(X1XY_train,Y1XY_train,"1 X and Y.png")
Y_pred = model.predict(X1XY_test)
print("1XY confusion matrix ",confusion_matrix(Y1XY_test,Y_pred))
print()
print("1XY accuracy matrix ",accuracy_score(Y1XY_test,Y_pred))
print(feature_variables_group1.columns)



























