import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data=pd.read_csv("C:\\20131A05M7\\Downloads\\loan\\training_set.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data=data.drop(columns=['Loan_ID'],axis=1)
l1=['Gender','Married','Dependents','Education','Self_Employed']
for i in l1:
    data[i].fillna(data[i].mode()[0], inplace=True)
data.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
credit_hist_count=data['Credit_History'].value_counts().values
credit_hist_index=data['Credit_History'].value_counts().index
plt.figure(figsize=(4,5))
sns.barplot(x=credit_hist_index,y=credit_hist_count,data=data)
Loan_Amount_Term_count=data['Loan_Amount_Term'].value_counts().values
Loan_Amount_Term_index=data['Loan_Amount_Term'].value_counts().index
plt.figure(figsize=(10,5))
sns.barplot(x=Loan_Amount_Term_index,y=Loan_Amount_Term_count,data=data)
ll=['Loan_Amount_Term','Credit_History']
for i in ll:
    data[i].fillna(data[i].mode()[0], inplace=True)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
l=["ApplicantIncome","CoapplicantIncome","LoanAmount"]
plt.figure(figsize=(12,6))
x=1
for i in l:
    plt.subplot(1,3,x)
    sns.boxplot(x=i,data=data)
    x = x + 1
plt.tight_layout()
for i in l:
    data[i].fillna(data[i].median(), inplace=True)
plt.figure(figsize=(12,6))
x=1
for i in ['Gender','Married','Dependents','Education','Self_Employed','property_Area']:
    plt.subplot(3,3,x)
    sns.countplot(x=i,data=data)
    x = x + 1
plt.tight_layout()
plt.figure(figsize=(12,6))
x=1
for i in ['Gender','Married','Dependents','Education','Self_Employed','property_Area']:
    plt.subplot(3,3,x)
    sns.countplot(x=i,hue="Loan_Status",data=data)
    x = x + 1
plt.tight_layout()
import seaborn as sns
sns.heatmap(data.corr(),annot=True,cmap="viridis")
for i in data:
    categorical_cols=data.select_dtypes(include=['object']).columns.tolist()
    numeric_cols=data.select_dtypes(include=['float']).columns.tolist()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data[['Gender','Married','Dependents','Education','Self_Employed','property_Area','Loan_Status']] = data[['Gender','Married','Dependents','Education','Self_Employed','property_Area','Loan_Status']].apply(LabelEncoder().fit_transform)
plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap="viridis")
data['Total_Income']=data['ApplicantIncome']+data['CoapplicantIncome']
data=data.drop(columns=["ApplicantIncome","CoapplicantIncome"],axis=1)
plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap="viridis")
X=data[['Total_Income','Credit_History','LoanAmount']]
Y=data['Loan_Status']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(X)
ros=RandomOverSampler(random_state=42)
x_res,y_res=ros.fit_resample(X,Y)
xtrain,xtest,ytrain,ytest=train_test_split(x_res,y_res,test_size=0.2,random_state=42)
rfc1=RandomForestClassifier()
rfc1.fit(xtrain,ytrain)
pickle.dump(rfc1,open('loan.pkl','wb'))
model=pickle.load(open('loan.pkl','rb'))
ypred=rfc1.predict(xtest)
print(classification_report(ypred,ytest))





