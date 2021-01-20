#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle


# In[2]:


#************************************************
#importing the dataset
#************************************************
df = pd.read_csv("F:/DATASCIENCE/Data/autos.csv", header=0, sep=',', encoding='Latin1',)


# In[3]:


#now printing the coulmn name and the shape of the dataset 
print(df.columns ,df.shape)


# In[4]:


#************************************************
#preprossesing and the data cleaning
#************************************************


# In[5]:


#printing the information of the salers
print(df.seller.value_counts())


# In[6]:


#removing the details of the saler who has only three cars
df[df.seller != 'gewerblich']


# In[7]:


# we have all the details of saller are same so we have to remove the column 
df=df.drop('seller',1)


# In[8]:


#print all the different sellers
print(df.offerType.value_counts())


# In[9]:


#remove the Offer Type having only 12 listings
df[df.offerType != 'Gesuch']


# In[10]:


#now all the offers are same so we can get rid of this column
df=df.drop('offerType',1)


# In[13]:


# Cars having power less than 50ps and above 900ps are little bit suspious so we have to get ride of thes column 
print(df.shape)


# In[14]:


df = df[(df.powerPS > 50) & (df.powerPS < 900)]
print(df.shape)


# In[15]:


#simlarly, filtering our the cars having registeration years not in the mentioned range
print(df.shape)


# In[16]:


df = df[(df.yearOfRegistration >= 1950) & (df.yearOfRegistration < 2017)]
print(df.shape)


# In[17]:


#the details which are irrelivent should be removed 
df.drop(['name','abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen',
         'postalCode', 'dateCreated'], axis='columns', inplace=True)


# In[18]:


#printing the final dataset 
print(df.shape)


# In[19]:


#final data that we have now
print(df.shape)


# In[20]:


#dropping the duplicates from the dataframe and stroing it in a new df.
#here all rows having same value in all the mentioned columns will be deleted and by default,
#only first occurance of anysuch row is kept
new_df = df.copy()
new_df = new_df.drop_duplicates(['price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])


# In[21]:


#after removing duplicates
print(new_df.shape)


# In[22]:


#As the dataset contained some german words for many features, cahnging them to english
new_df.gearbox.replace(('manuell', 'automatik'), ('manual','automatic'), inplace=True)
new_df.fuelType.replace(('benzin','andere','elektro'),('petrol','others','electric'),inplace=True)
new_df.vehicleType.replace(('kleinwagen', 'cabrio','kombi','andere'),
                           ('small car','convertible','combination','others'),inplace=True)
new_df.notRepairedDamage.replace(('ja','nein'),('Yes','No'),inplace=True)


# In[23]:


#### Removing the outliers
new_df = new_df[(new_df.price >= 100) & (new_df.price <= 150000)]


# In[24]:


#Filling NaN values for columns whose data might not be there with the information provider,
#which might lead to some variance but our model
#but we will still be able to give some estimate to the user
new_df['notRepairedDamage'].fillna(value='not-declared', inplace=True)
new_df['fuelType'].fillna(value='not-declared', inplace=True)
new_df['gearbox'].fillna(value='not-declared', inplace=True)
new_df['vehicleType'].fillna(value='not-declared', inplace=True)
new_df['model'].fillna(value='not-declared', inplace=True)


# In[25]:


#can save the csv for future purpose. 
new_df.to_csv("autos_preprocessed.csv")


# In[26]:


#Columns which contain categorical values, which we'll need to convert via label encoding
labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']


# In[27]:


#looping over the labels to do the label encoding for all at once and 
#saving the LABEL ENCODING FILES
mapper = {}
for i in labels:
    mapper[i] = LabelEncoder()
    mapper[i].fit(new_df[i])
    tr = mapper[i].transform(new_df[i])
    np.save(str('classes'+i+'.npy'), mapper[i].classes_)
    print(i,":",mapper[i])
    new_df.loc[:, i + '_labels'] = pd.Series(tr, index=new_df.index)


# In[28]:


#Final data to be put in a new dataframe called "LABELED",
labeled = new_df[ ['price'
                        ,'yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        ] 
                    + [x+"_labels" for x in labels]]

print(labeled.columns)


# In[29]:


#**********************************************************
##Train-Test Split
#**********************************************************


# In[30]:


#Storing price in Y and rest of the data in X
Y = labeled.iloc[:,0].values
X = labeled.iloc[:,1:].values


# In[31]:


#need to reshape the Y values
Y = Y.reshape(-1,1)


# In[32]:


from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 3)


# In[33]:


#**********************************************************
##Model building and Fitting
#**********************************************************


# In[34]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,random_state=34)


# In[38]:


#fitting the model
regressor.fit(X_train, np.ravel(Y_train,order='C'))


# In[39]:


#predicting the values fo test test
y_pred = regressor.predict(X_test)


# In[40]:


#printing the Accuraccy for test set
print(r2_score(Y_test,y_pred))


# In[42]:


#for testing on user input values
y_pred1 = regressor.predict([[2011,190,125000,5,1,0,163,1,3,3]])


# In[43]:


#predticting price for a user input values
print(y_pred1)


# In[44]:


#saving the model for future use.
filename = 'resale_model.sav'
pickle.dump(regressor, open(filename, 'wb'))


# In[ ]:




