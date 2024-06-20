#########CLEANING DATASET#########  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_excel('data.xlsx')

#check for duplicates and display them, then drop duplicates
duplicates_rows = df[df.duplicated()]
print("Duplicate rows:")
print(duplicates_rows)

#removing duplicates
df.drop_duplicates(inplace=True)

# Print the DataFrame to confirm duplicates are removed
print("DataFrame after removing duplicates:")
print(df.to_string())

# replacing empty cells with median where axis = 0
# Specify specific columns for which you want to replace missing values with median
columns_to_fill = ['radius_mean','texture_mean','perimeter_mean','area_mean',
'smoothness_mean','compactness_mean','concavity_mean','concave_points_mean',
'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
'area_se','smoothness_se','compactness_se','concavity_se','concave_points_se',
'symmetry_se','fractal_dimension_se','radius_worst','texture_worst',
'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
'concavity_worst','concave_points_worst','symmetry_worst',
'fractal_dimension_worst']

# Calculate median for the specified columns
median_values = df[columns_to_fill].median() 
df[columns_to_fill] = df[columns_to_fill].fillna(median_values)
print("DataFrame after replacing empty cells with median along columns:")
print(df.to_string())

#describe the dataset
print("Describing the dataset")
print(df.describe())



# #####machine learning######




#split dataset into training set and test set
x=df.drop(columns=['id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean',
'smoothness_mean','compactness_mean','concavity_mean','concave_points_mean',
'symmetry_mean','fractal_dimension_mean'])
y=df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#create a decision tree, logistic regression, svm, random forest classifiers
Decision_tree_model=DecisionTreeClassifier()
Logistic_regression_model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

#train the models using the training sets
Decision_tree_model.fit(x_train, y_train)
Logistic_regression_model.fit(x_train, y_train)
SVM_model.fit(x_train, y_train)
RF_model.fit(x_train, y_train)

#predict the models
DT_prediction=Decision_tree_model.predict(x_test)
LR_prediction=Logistic_regression_model.predict(x_test)
SVM_prediction=SVM_model.predict(x_test)
RF_prediction=RF_model.predict(x_test)

#calculation of model accuracy score for each model
DT_score=accuracy_score(y_test,DT_prediction)
LR_score=accuracy_score(y_test,LR_prediction)
SVM_score=accuracy_score(y_test,SVM_prediction)
RF_score=accuracy_score(y_test,RF_prediction)

#display accuracy score for each model
print("Decision Tree Accuracy=", DT_score*100,"%")
print("Logistic Regression Accuracy=", LR_score*100,"%")
print("Support Vector Machine Accuracy=", SVM_score*100,"%")
print("Random Forest Accuracy=", RF_score*100,"%")


##After viewing each model accuracy score, you may use one which
##suits your specific requirements of your problem
##in our case [ we use svm = 97.36842105263158 % ]


##creating a persisting model

#learning with svm
model=svm.SVC(kernel='linear')
model.fit(x.values,y)

#create a persisting model
joblib.dump(model, 'cancer-recommender.joblib')

##use the created persisting model

#user inputs


# radius_mean=int(input("Enter radius_mean:"))
# texture_mean=int(input("Enter radius_mean:")) 
# perimeter_mean=int(input("Enter radius_mean:"))
# area_mean=int(input("Enter radius_mean:"))
# smoothness_mean=int(input("Enter radius_mean:"))
# compactness_mean=int(input("Enter radius_mean:")) 
# concavity_mean=int(input("Enter radius_mean:")) 
# concave_points_mean=int(input("Enter radius_mean:"))
# symmetry_mean=int(input("Enter radius_mean:"))
# fractal_dimension_mean=int(input("Enter radius_mean:")) 

radius_se = float(input("Enter radius se:"))
texture_se = float(input("Enter texture se:"))
perimeter_se = float(input("Enter perimeter se:"))
area_se = float(input("Enter area se:")) 
smoothness_se = float(input("Enter smoothness se:")) 
compactness_se = float(input("Enter compactness se:"))
concavity_se = float(input("Enter concavity se:"))
concave_points_se = float(input("Enter concave points se:"))
symmetry_se = float(input("Enter symmetry se:")) 
fractal_dimension_se = float(input("Enter fractal dimension se:")) 
radius_worst = float(input("Enter radius worst:")) 
texture_worst = float(input("Enter texture worst:"))
perimeter_worst = float(input("Enter perimeter worst:"))
area_worst = float(input("Enter area worst:"))
smoothness_worst = float(input("Enter smoothness worst:")) 
compactness_worst = float(input("Enter compactness worst:"))
concavity_worst = float(input("Enter concavity worst:")) 
concave_points_worst = float(input("Enter concave points worst:"))
symmetry_worst = float(input("Enter symmetry worst:"))
fractal_dimension_worst = float(input("Enter fractal dimension worst:"))


# #predict with the created model
model=joblib.load('cancer-recommender.joblib')
predictionss=model.predict([[radius_se,texture_se,perimeter_se,
area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
symmetry_se,fractal_dimension_se,radius_worst,texture_worst,
perimeter_worst,area_worst,smoothness_worst,compactness_worst,
concavity_worst,concave_points_worst,symmetry_worst,
fractal_dimension_worst]])

# predictions = model.predict([[1.095,0.9053,8.589,153.4,0.0064,0.04904,0.05373,0.01587
# ,0.03003,0.00619,25.38,17.33,184.6,2019,0.1622,0.7119,0.2654,0.4601,0.1189]])
print("Diagnosis of Breast Tissues : ",predictionss)

