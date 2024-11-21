#!/usr/bin/env python
# coding: utf-8

# # All Machine learning algorithms

# In[71]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm


# In[72]:


train = pd.read_csv("C:/Users/govar/Downloads/train_v9rqX0R (1).csv")


# In[73]:


train.head()


# In[74]:


## Please load test dataset and perform same operatins on it
test = pd.read_csv("C:/Users/govar/Downloads/test_AbJTz2l (1).csv")


# In[75]:


test.head()


# In[76]:


# Print Dimensions of the dataset
train.shape


# In[77]:


test.shape


# In[78]:


# Summary of the dataset
train.info()


# In[79]:


train.isnull().sum()


# In[80]:


# check for duplicate values
duplicate =  train.duplicated().sum()

print(f"Number of duplicate entries: {duplicate}")


# # Univariant analysis

# In[81]:


# Distribution of Target Variable
sns.histplot(train['Item_Outlet_Sales'], kde = True)
plt.show()

# Distribtution of numerial distributions
sns.histplot(train['Item_Weight'].dropna(), kde = True)
plt.show()
# Please check the distribution of rest of the numerical features...
sns.histplot(train['Outlet_Establishment_Year'].dropna(), kde = True)
plt.show()

sns.histplot(train['Item_MRP'].dropna(), kde = True)
plt.show()

sns.histplot(train['Item_Visibility'].dropna(), kde = True)
plt.show()


# In[84]:


# Count Categorical Features
train['Item_Fat_Content'].value_counts()

# .... Continue


# In[86]:


train.head()


# In[88]:


# Check unique values in the column
print(train['Item_Fat_Content'].unique())

# Handling inconsistent values
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'Low Fat': 'Low Fat',
    'reg': 'Regular',
    'RG': 'Regular',
    'Regular': 'Regular'
})


# In[90]:


train.head()


# In[91]:


train['Item_Fat_Content'].value_counts()


# In[93]:


train['Item_Identifier'].value_counts()


# In[100]:


train['Item_Type'].value_counts()


# In[101]:


train['Outlet_Identifier'].value_counts()



# In[102]:


train['Outlet_Size'].value_counts()


# In[103]:


train['Outlet_Location_Type'].value_counts()


# In[105]:


train.isnull().sum()


# # Data preprocessing

# In[106]:


# Handling Missing Values
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace = True) # Check the use of inplace parameter



# In[107]:


train['Item_Weight'].isnull().sum()


# In[108]:


train['Outlet_Size'].mode()


# In[109]:


train['Outlet_Size'].fillna('Medium', inplace=True)


# In[113]:


train['Outlet_Size'].isnull().sum()


# In[115]:


train.head()


# In[116]:


train.isnull().sum()


# In[117]:


train.head()


# In[119]:


train['outlet_age']


# In[120]:


# Transform features (outlet_age)
train['outlet_age'] = 2024 - train['Outlet_Establishment_Year']


# In[121]:


train['outlet_age'].head()


# # Bivariant Analysis

# In[122]:


# Relationship between numerical features and target values

sns.scatterplot(data=train, x='Item_Weight', y = 'Item_Outlet_Sales')
plt.show()

sns.scatterplot(data=train, x='Item_MRP', y = 'Item_Outlet_Sales')
plt.show()

# Continue ploting for rest of the features


# In[29]:


sns.boxplot(data=train, x = 'Item_Fat_Content', y = 'Item_Outlet_Sales')
plt.show()


# In[30]:


sns.boxplot(data=train, x = 'Item_Type', y = 'Item_Outlet_Sales')
plt.show()


# In[31]:


sns.boxplot(data=train, x = 'Outlet_Size', y = 'Item_Outlet_Sales')
plt.show()


# In[32]:


train.head(2)


# In[ ]:





# # multivariant Analysis

# In[33]:


# Interaction between features
sns.barplot(data=train, x = 'Item_Outlet_Sales', y = 'Item_Type', hue = 'Outlet_Type')
plt.show()

sns.barplot(data = train, x='Item_Outlet_Sales', y='Item_Fat_Content', hue='Item_Fat_Content')
plt.show()
# .... Continue


# In[34]:


sns.boxplot(train['Item_Outlet_Sales'])
plt.show()

# Handling outliers
# Can transform, remove, cap the outlier


# Removing outliers using box-plot

# # IQR method

# In[123]:


q1 = train['Item_Outlet_Sales'].quantile(0.25)
q3 = train['Item_Outlet_Sales'].quantile(0.75)
iqr = q3-q1


# In[124]:


q1,q3,iqr


# In[127]:


upper_limit = q3 + (1.5*iqr)
lower_limit = q1 - (1.5*iqr)
lower_limit, upper_limit


# Before removing outliers

# In[126]:


sns.boxplot(train['Item_Outlet_Sales'])


# In[ ]:





# In[128]:


train.loc[(train['Item_Outlet_Sales'] > upper_limit) | (train['Item_Outlet_Sales'] < lower_limit)]


# In[130]:


new_train = train.loc[(train['Item_Outlet_Sales'] < upper_limit) & (train['Item_Outlet_Sales'] > lower_limit)]
print('before removing outliers:', len(train))
print('after removing outliers:', len(new_train))
print('outliers:', len(train)-len(new_train))


# In[131]:


sns.boxplot(new_train['Item_Outlet_Sales'])


# In[132]:


train['Outlet_Location_Type'].unique()


# In[133]:


train['Outlet_Type'].unique()


# # Feature Engineering

# In[134]:


# New feature for price range
train['Price_Rage'] = pd.qcut(train['Item_MRP'], q = 4) # Use google to find out what this function does.

# Create dummies for categorical variabl
dummy_columns = [col for col in train.columns if col.startswith('Item_Identifie')]

# Drop the dummy columns
train = train.drop(dummy_columns, axis=1)


# In[135]:


train = train.drop('Outlet_Identifier', axis=1)


# In[136]:


train = train.drop('Item_Type', axis=1)


# In[137]:


train = pd.get_dummies(train, columns=['Outlet_Size'])


# In[138]:


train = pd.get_dummies(train, columns=['Outlet_Location_Type'])


# In[140]:


# Create dummies for categorical variabl
dummy_columns1 = [col for col in train.columns if col.startswith('Item_Fat_Content')]

# Drop the dummy columns
train = train.drop(dummy_columns1, axis=1)


# In[141]:


train['Outlet_Size_High'] = train['Outlet_Size_High'].astype(int)


# In[142]:


train['Outlet_Size_Medium'] = train['Outlet_Size_Medium'].astype(int)


# In[143]:


train['Outlet_Size_Small'] = train['Outlet_Size_Small'].astype(int)


# In[144]:


train['Outlet_Location_Type_Tier 1'] = train['Outlet_Location_Type_Tier 1'].astype(int)


# In[145]:


train['Outlet_Location_Type_Tier 2'] = train['Outlet_Location_Type_Tier 2'].astype(int)


# In[146]:


train['Outlet_Location_Type_Tier 3'] = train['Outlet_Location_Type_Tier 3'].astype(int)


# In[147]:


train.head()


# In[148]:


train1 = train.drop('Price_Rage', axis=1)


# In[149]:


train =train.drop('Outlet_Type', axis=1)


# In[150]:


train1 = train1.drop('Outlet_Type', axis=1)


# # Visualizations

# In[151]:


## Correlation plot
corr = train1.corr()
sns.heatmap(corr, annot=True, cmap = 'coolwarm')
plt.show()


# # Reporting

# In[152]:


# Summarize your findings. Do not use code snipets. Please write at least 2 lines about each feature.
Key observations from the matrix:

Strong correlations: Some variables are strongly correlated like Outlet_Establishment_Year and outlet_age are perfectly negatively correlated, which makes sense as they represent the same information in different formats.
    
Moderate correlations: Some variables have moderate correlations those are Item_MRP seems to have a moderate positive correlation with Item_Outlet_Sales, suggesting that higher priced items tend to have higher sales.
    
Weak correlations: Some variables have weak or no correlation. This indicates that these variables might not be very influential in predicting Item_Outlet_Sales.


# # Fitting a model (Linear Regression)

# In[153]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm 


# In[154]:


x = train1.drop(columns='Item_Outlet_Sales')
y = train1['Item_Outlet_Sales']


# In[155]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=141)


# In[156]:


x_train.head()


# In[157]:


x


# In[ ]:





# In[158]:


x_test.shape


# In[159]:


y_train.shape


# In[160]:


y_test.shape


# In[161]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[162]:


x_train


# In[163]:


x_test


# In[164]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)


# In[165]:


y_pred = regression.predict(x_test)


# In[166]:


from sklearn.metrics import r2_score


# In[167]:


r2_score(y_test, y_pred)


# # Random forest regression

# In[168]:


from sklearn.ensemble import RandomForestRegressor
regression_rf = RandomForestRegressor()
regression_rf.fit(x_train, y_train)


# In[169]:


y_pred = regression_rf.predict(x_test)


# In[170]:


r2_score(y_test, y_pred)


# In[171]:


train1.head()


# In[172]:


frank = [[9.30, 0.016047, 249.8092, 1999, 3735.1380, 25, 0, 1, 0, 1, 0]]


# In[173]:


regression_rf.predict(sc.transform(frank))


# In[174]:


# Assumption 1: Linearity
sns.scatterplot(x = y_test, y = y_pred)
plt.Xlable('actual sales ')
plt.Ylable('predited sales ')
plt.title ('actual vs prdeicted sales')
plt.show()


# In[178]:


# Checking assumptions


# Assumption 1: Linearity
sns.scatterplot(x = y_test, y = y_pred)
plt.xlable('actual sales ')
plt.ylable('predited sales ')
plt.title ('actual vs prdeicted sales')
plt.show()

# Assumption  2 : Normaity of residuals
residuals = _test - y_pred
sns.histplot(residuals, kde = Ture)
plt.title('Residuals distribution')
plt.show()

# Assumption 3 : Homoscedasticity
sns.scatterplot(x = y_pred, y = residuals)
plt.xlabel('Predicted sales')
plt.ylabel('Residuals')
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.show()



# Assumption 4: INdependence of residuals
# durbin_watson test


# Assumption 5: No multicollinearitty among predictors
vif_data = pd.DataFrame()
vif_data['features' ] = x_train.columns
vif_data['VIF'] = [variance_inflation_factor(x_train.values,i) for i in range(len(x_train.columns))]
vif_data


# In[ ]:





# In[181]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
# You can specify hyperparameters like max_depth, min_samples_split, etc.
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




