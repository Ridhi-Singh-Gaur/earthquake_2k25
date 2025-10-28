#!/usr/bin/env python
# coding: utf-8

# In[2]:


#In this project we are predicting POSSIBILITY(in the form of 0 or 1) & the MAGNITUDE of AN EARTHQUAKE.------+++
#our model will give 1 as the outcome for earthquake will occur and 0 as the outcome for the earthquake will not occur. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,classification_report,roc_curve,auc


# In[3]:


data=pd.read_csv("C://Users//ridhi//Downloads//query (1) (1).csv",encoding='latin-1')


# In[4]:


data


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


#some of the columns in this dataset are not beneficial for predicting our output so removing them.
data=data.drop(["id","updated","status","locationSource","magSource","depthError","magError","magNst","horizontalError","type"],axis=1)
data


# In[9]:


#adding a column earthquake which has value as 0 when magnitude of earthquake is less than 3 and it has value 1 when magnitude is greater than or equal to 3.
data['earthquake']=np.where(data['mag']>=3,1,0)
data


# In[10]:


data['earthquake'].value_counts()


# In[11]:


data.isnull()


# In[12]:


#finding the no of null values in each column
data.isnull().sum()


# In[13]:


#nst,gap,dmin are the columns with the null values.
#finding the datatype of each of them
print("nst has dtype ",data["nst"].dtype)
print("gap has dtype ",data["gap"].dtype)
print("dmin has dtype ",data["dmin"].dtype)


# In[14]:


#since all of the columns having null values are numeric 
#therefore replacing the null values with the mean of each of the columns
data["nst"].fillna(data["nst"].mean(),inplace=True)
data["gap"].fillna(data["gap"].mean(),inplace=True)
data["dmin"].fillna(data["dmin"].mean(),inplace=True)


# In[15]:


print(data["nst"].isnull().sum()," ",data["gap"].isnull().sum()," ",data["dmin"].isnull().sum())


# In[16]:


#working with the dates 
#converting the data type of date_time column from object to datetime
data['time']=pd.to_datetime(data['time'])
data["month"]=data["time"].dt.month
#dropping the time column from the dataset
data=data.drop("time",axis=1)
data


# In[17]:


#extracting the exact places which are affected
data['place'] = data['place'].apply(lambda x: x.split(', ')[1] if ', ' in x else x)
data['place']


# In[18]:


#visualising our dataset
data["place"].value_counts()


# In[19]:


series3=data["mag"].value_counts()
x2=np.array(series3.index)
y2=series3.values
print("datatype of x2 ",type(x2))
print("datatype of y2 ",type(y2))
print("\n")
fig=plt.figure(figsize=(5,5))
plt.hist(x2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("frequency of earthquake",fontsize=10)
plt.title("a plot to show the frequency of each magnitude of earthquake present in dataset",fontsize=10)
plt.xlabel("magnitude of earthquake",fontsize=10)
plt.show()


# In[20]:


series2=data["place"].value_counts()
x1=np.array(series2.index)
y1=series2.values
print("datatype of x1 ",type(x1))
print("datatype of y1 ",type(y1))
print("\n")
fig=plt.figure(figsize=(30,70))
plt.barh(x1,y1)
plt.xticks(fontsize=30)
plt.yticks(fontsize=15)
plt.ylabel("name of place",fontsize=30)
plt.title("a plot to show the frequency of months which are taken for the prediction",fontsize=30)
plt.xlabel("frequency",fontsize=30)
plt.show()


# In[21]:


#visualising correlations
fig=plt.figure(figsize=(10,10))
correlation_matrix=data.corr(numeric_only=True)
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.show()


# In[22]:


data


# In[23]:


data.columns


# In[24]:


data=data[['month','latitude', 'longitude', 'depth', 'magType', 'nst', 'gap','dmin', 'rms', 'net', 'place', 'earthquake','mag']]


# In[25]:


data.info()


# In[26]:


#Encoding the categorical data ie converting datatypes of those columns which have datatypes as object into integer
#using label encoding 
label_encoders={}
categorical_columns=['net','magType','place',]
for column in categorical_columns:
    label_encoders[column]=LabelEncoder()
    data[column]=label_encoders[column].fit_transform(data[column])
data


# In[27]:


#training our model and predicting the outcome
X=data.iloc[:,:-2].values
Y=data.iloc[:,-2].values


# In[28]:


Y


# In[29]:


#feature engineering
#feature selection 
from sklearn.feature_selection import SelectKBest,f_classif
selector=SelectKBest(f_classif,k=11)
X_new=selector.fit_transform(X,Y)
print(X_new)


# In[30]:


#Feature scaling
scaler=StandardScaler()
scaled_features=scaler.fit_transform(X_new)
scaled_features


# In[31]:


#Splitting the dataset
X_train,X_test,Y_train,Y_test=train_test_split(scaled_features,Y,test_size=1/3,random_state=0)
#CLASSIFICATION ALGORITHMS
classifiers={
    'Logistic Regression':LogisticRegression(),
    'Support Vector Machine ':SVC(),
    'Decision Tree':DecisionTreeClassifier(),
    'Random Forest':RandomForestClassifier(),
    'Naive Bayes':GaussianNB(),
    'K Nearest Neighbour':KNeighborsClassifier()
}


# In[32]:


#Training and evaluating classifiers
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
results={}
for name,clf in classifiers.items():
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    cm=confusion_matrix(Y_test,Y_pred)
    print(f"Confusion matrix for {name} is \n",cm)
    accuracy=accuracy_score(Y_test,Y_pred)
    results[name]=accuracy
    print(f"{name} has accuracy of {accuracy*100:.2f} ")
    print(classification_report(Y_test,Y_pred,zero_division=1))
    print("\n\n")
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(cm,fmt='d',annot_kws={"size": 14}, annot=True,cmap='coolwarm')
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('predicted labels')
    plt.xlabel('true labels')
    plt.show


# In[33]:


#finding the best classifier
best_classifier=max(results,key=results.get)
print("best classifier is ",best_classifier," with an accuracy of ",results[best_classifier])


# In[34]:


fpr,tpr,thresholds=roc_curve(Y_test,Y_pred)

#calculating the auc
roc_auc=auc(fpr,tpr)
roc_auc_rounded = round(roc_auc, 2)
#plot the roc curve
plt.figure(figsize=(4,4))
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve(AUC area={roc_auc_rounded}')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('Reciever operating characteristic curve')
plt.legend(loc='best')
plt.show()


# In[35]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train the model
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, Y_train)

# Print the number of features and classes to verify
print("Number of features in model:", model.n_features_in_)
print("Number of classes in model:", len(model.classes_))

# Plot the decision tree
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size as needed
plot_tree(
    model,
    feature_names=['month', 'latitude', 'longitude', 'depth', 'magType', 'nst', 'gap', 'dmin', 'rms', 'net', 'place'],
    class_names=['0','1'],
    filled=True,
    rounded=True,
    fontsize=13,  # Font size for text
    ax=ax
)

# Customize plot appearance
plt.title("Simplified Decision Tree Visualization")
ax.set_aspect('auto')  # Ensure aspect ratio is equal

# Show the plot
plt.show()


# In[36]:


#predicting the magnitude of earthquake 
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[37]:


#feature engineering
#feature selection 
from sklearn.feature_selection import SelectKBest,f_classif
selector1=SelectKBest(f_classif,k=12)
x_new=selector1.fit_transform(x,y)
print(x_new)


# In[38]:


#Feature scaling
scaler1=StandardScaler()
scaled_features1=scaler.fit_transform(x_new)
scaled_features1


# In[39]:


#Splitting the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[40]:


#here multilinear regression is used.
regressor=LinearRegression()


# In[41]:


regressor.fit(x_train,y_train)


# In[42]:


y_predictAll=regressor.predict(x)
y_predictAll


# In[43]:


#finding the regression coefficients
regressor.coef_


# In[44]:


#finding the intercepts
regressor.intercept_


# In[45]:


#model evaluation
#printing the r squared
print("r squared ",r2_score(y,y_predictAll)*100)


# In[46]:


#printing the mean squared error
print("mean squared error is ",mean_squared_error(y,y_predictAll))


# In[47]:


regressor.score(x,y)*100


# In[48]:


regressor.score(x_train,y_train)*100


# In[49]:


regressor.score(x_test,y_test)*100


# In[50]:


#as our model gives a score of 86.82 % score on the testing data which is very good


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




