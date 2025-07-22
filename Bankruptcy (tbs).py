#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_excel('Bankruptcy.xlsx')   # loading dataset
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# ### Exploratory Data Analysis

# In[6]:


df.isnull().sum() # no missing values


# In[7]:


df.duplicated().sum()   # duplicates identified


# In[8]:


df.drop_duplicates(inplace=True)  #  duplicates removed
df.duplicated().sum()


# In[9]:


plt.figure(figsize=(10,6))  # no outliers identified
sns.boxplot(df.drop(columns=['class']))
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()


# In[10]:


corr= df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


# ### Model Building

# In[12]:


df['class'].unique()  # target columns


# In[13]:


label= LabelEncoder()   # encoding target column using Labelencoder
df['class']= label.fit_transform(df['class'])
df['class'].unique()


# In[14]:


X= df.drop(columns=['class'])  # features  
y= df['class']  # target 


# In[15]:


X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.8,shuffle=True,random_state=42)  # splitting dataset


# In[16]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


scaler= StandardScaler()                   # Feature scaling using StandardScaler
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# #### Model Training & Evaluation

# In[19]:


original_models= {'Logistic Regression':LogisticRegression(),
                  'K-Nearest Neighbors': KNeighborsClassifier(),
                  'Decision Tree': DecisionTreeClassifier(),
                  'Random Forest': RandomForestClassifier(),
                  'Gradient Boosting': GradientBoostingClassifier(),
                 'XG Boosting': XGBClassifier()}


# In[20]:


for name,model in original_models.items():
    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    
    print(f'{name} Accuracy: {accuracy_score(y_test,y_pred):.2f}')
    print(classification_report(y_test,y_pred))


# ### Model Deployment using Streamlit

# In[22]:


import streamlit as st
import pickle


# In[23]:


file = 'bankrupt.pkl'
pkl = pickle.dump(original_models, open(file, 'wb'))


# In[24]:


model = pickle.load(open(file, 'rb'))['K-Nearest Neighbors']    # KNN model with highest accuracy


# In[25]:


st.title('Bankruptcy Prediction App')


# In[26]:


st.sidebar.header('Enter Company Financial Data')
features = ['Industrial Risk', 'Management Risk', 'Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk']
user_data = {}
for feature in features:
    user_data[feature] = st.sidebar.selectbox(feature, options=[0,0.5,1],index=0)


# In[27]:


input_data = pd.DataFrame(user_data, index=[0])
rename_dict= {'Industrial Risk': 'industrial_risk',
              'Management Risk': 'management_risk',
              'Financial Flexibility':'financial_flexibility',
              'Credibility': 'credibility',
              'Competitiveness': 'competitiveness',
              'Operating Risk': 'operating_risk'}


# In[28]:


input_data.rename(columns=rename_dict, inplace=True)
input_data= input_data[rename_dict.values()]


# In[29]:


input_scaled= scaler.transform(input_data)    # standardscaler


# In[30]:


if st.sidebar.button('Predict Bankruptcy'):
    prediction = model.predict(input_scaled)
    prediction_prob = model.predict_proba(input_scaled)
    
    if prediction[0] == 1:
        st.sidebar.error(f'The company is likely to go bankrupt with {prediction_prob[0][1] * 100:.2f}% probability.')
    else:
        st.sidebar.success(f'The company is NOT likely to go bankrupt with {prediction_prob[0][0] * 100:.2f}% probability.')
    st.write('### Bankruptcy Prediction Probability')
    fig, ax = plt.subplots()
    labels = ['Not Bankrupt', 'Bankrupt']
    ax.bar(labels, prediction_prob[0], color=['green', 'red'])
    ax.set_ylabel('Probability')
    st.pyplot(fig)


# In[61]:


st.write('### Preview of the Dataset')
st.dataframe(df.head())
st.write('Statistical Summary:')
st.dataframe(df.describe())

