# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data.

3. Use modelselection and Countvectorizer to preditct the values.

4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:Darshan V
RegisterNumber:212224230050
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("NAME:SHIVANI M")
print("Register No: 212224040313")
print(classification_report1)


```

## Output:
#### DATA
<img width="793" height="218" alt="image" src="https://github.com/user-attachments/assets/2c5745e7-edeb-4d66-b495-c0f0e2bc391c" />

#### X TRAIN
<img width="1362" height="213" alt="image" src="https://github.com/user-attachments/assets/0e1c38b5-86be-4e5b-9846-e42783e843f7" />

#### X TEST
<img width="1352" height="261" alt="image" src="https://github.com/user-attachments/assets/db13e5af-f7b3-4681-812a-4c90d36590f8" />

#### ACCURACY

<img width="238" height="48" alt="image" src="https://github.com/user-attachments/assets/066bd6b7-015f-4c4c-a698-44744394fab2" />

#### CONFUSION MATRIX

<img width="376" height="71" alt="image" src="https://github.com/user-attachments/assets/3268f755-c6a9-4539-837e-4b77807cab1f" />

#### CLASSIFICATION REPORT
<img width="631" height="235" alt="image" src="https://github.com/user-attachments/assets/979b84f0-23b4-489e-aac4-e9802a4ca9d6" />





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
