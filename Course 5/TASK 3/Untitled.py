
# coding: utf-8

# In[ ]:


####################################################################################
# Title -- Course 5 Task 3 -- 
# Step 1 -- Prepare and Explore the Data
# Step 2 -- Build and Evaluate the Models

# Last Updated -- 10 Dec 19

# File:  
# Step 1 -- Course 5 Task 2
# Step 2 -- Course 5 Task 3 (which will include Step 1)

# Project Name:  Credit One Customer Default Rates Analysis
####################################################################################


# In[ ]:


####################################################################################
# Project Notes
####################################################################################

# Summary of Project:  Credit One has seen an increase in the number of custormers that
# default on their credit card payments.  This is bad because Credit One is the company 
# that approves the customer loans for its clients.  This is also bad as it is a loss of 
# revenue for the clients and potentially Credit One, if the clients no longer trust
# Credit One's approval methodology.  The project attempts to identify significant patterns
# in current customers that default on their loans and to build a model to predict if future
# customers are likely to default on their loans.  This analysis should further inform the
# Credit One methodology on approving loans for their clients.

# Summarize the top model and/or filtered dataset


# In[ ]:


####################################################################################
# Housekeeping
####################################################################################


# In[ ]:


####################################################################################
# Load Packages
####################################################################################


# In[5]:


#imports
#numpy, pandas, scipy, math, matplotlib
import numpy as np
import pandas as pd #used for importing data
import scipy
from math import sqrt
import matplotlib.pyplot as plt

#estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model

#model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve,GridSearchCV

#cross validation
from sklearn.model_selection import train_test_split

# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[222]:


#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[ ]:


#set working directory as necessary
#os.chdir('')


# In[7]:


# Get working directory
os.getcwd()


# In[ ]:


####################################################################################
# Import Data
####################################################################################


# In[8]:


# Load Raw datasets or Train/Test Dataset -- Dataset 1
#### Datasets for Analysis
#### Datasets for Analysis
credit_OOB = pd.read_csv('credit_OOB.csv', header =0) #out of box with ID removed and Education regrouped (OTHER)
credit_OOB_S = pd.read_csv('Credit_OOB_S.csv', header =0) #out of box scaled
credit_OOB_FS = pd.read_csv('Credit_OOB_FS.csv', header =0) # out of box removed highly correlated
credit_OOB_FS_S = pd.read_csv('Credit_OOB_FS_S.csv', header =0) # out of box FS scaled
credit_OOB_RFE = pd.read_csv('Credit_OOB_RFE.csv', header =0) #OOB using RFE top 10
credit_OOB_RFE_S = pd.read_csv('Credit_OOB_RFE_S.csv', header =0) #OOB RFE scaled
credit_DV = pd.read_csv('Credit_DV.csv', header =0) #All categorical variables turned into Dummy Variables, age / lim binned
credit_DV_S = pd.read_csv('Credit_DV_S.csv', header =0) # DV Scaled
credit_DV_FS = pd.read_csv('Credit_DV_FS.csv', header =0) #DV with removed highly correlated
credit_DV_FS_S = pd.read_csv('Credit_DV_FS_S.csv', header =0) #DV_FS scaled
credit_DV_RFE = pd.read_csv('Credit_DV_RFE.csv', header =0) #DV using RFE top 10
credit_DV_RFE_S = pd.read_csv('Credit_DV_RFE_S.csv', header =0) #DV RFE scaled


# In[ ]:


####################################################################################
# Verify data input worked
####################################################################################


# In[9]:


credit_OOB.shape, credit_OOB_S.shape, credit_OOB_FS.shape, credit_OOB_FS_S.shape, credit_OOB_RFE.shape, credit_OOB_RFE_S.shape


# In[10]:


credit_DV.shape, credit_DV_S.shape, credit_DV_FS.shape, credit_DV_FS_S.shape, credit_DV_RFE.shape, credit_DV_RFE_S.shape


# In[11]:


# moving dependent variable to the end of the dataframe
credit_OOB = credit_OOB.drop(columns=['dpnm']).assign(dpnm=credit_OOB['dpnm'])
credit_OOB_S = credit_OOB_S.drop(columns=['dpnm']).assign(dpnm=credit_OOB_S['dpnm'])
credit_OOB_FS = credit_OOB_FS.drop(columns=['dpnm']).assign(dpnm=credit_OOB_FS['dpnm'])
credit_OOB_FS_S = credit_OOB_FS_S.drop(columns=['dpnm']).assign(dpnm=credit_OOB_FS_S['dpnm'])
credit_OOB_RFE = credit_OOB_RFE.drop(columns=['dpnm']).assign(dpnm=credit_OOB_RFE['dpnm'])
credit_OOB_RFE_S = credit_OOB_RFE_S.drop(columns=['dpnm']).assign(dpnm=credit_OOB_RFE_S['dpnm'])

credit_DV = credit_DV.drop(columns=['dpnm']).assign(dpnm=credit_DV['dpnm'])
credit_DV_S = credit_DV_S.drop(columns=['dpnm']).assign(dpnm=credit_DV_S['dpnm'])
credit_DV_FS = credit_DV_FS.drop(columns=['dpnm']).assign(dpnm=credit_DV_FS['dpnm'])
credit_DV_FS_S = credit_DV_FS_S.drop(columns=['dpnm']).assign(dpnm=credit_DV_FS_S['dpnm'])
credit_DV_RFE = credit_DV_RFE.drop(columns=['dpnm']).assign(dpnm=credit_DV_RFE['dpnm'])
credit_DV_RFE_S = credit_DV_RFE_S.drop(columns=['dpnm']).assign(dpnm=credit_DV_RFE_S['dpnm'])


# In[336]:


####################################################################################
# Select the Features
####################################################################################


# In[12]:


features_OOB = credit_OOB.iloc[:,0:23]
features_OOB_S = credit_OOB_S.iloc[:,0:23]
features_OOB_FS = credit_OOB_FS.iloc[:,0:20]
features_OOB_FS_S = credit_OOB_FS_S.iloc[:,0:20]
features_OOB_RFE = credit_OOB_RFE.iloc[:,0:10]
features_OOB_RFE_S = credit_OOB_RFE_S.iloc[:,0:10]


# In[13]:


features_DV = credit_DV.iloc[:,0:86]
features_DV_S = credit_DV_S.iloc[:,0:86]
features_DV_FS = credit_DV_FS.iloc[:,0:80]
features_DV_FS_S = credit_DV_FS_S.iloc[:,0:80]
features_DV_RFE = credit_DV_RFE.iloc[:,0:10]
features_DV_RFE_S = credit_DV_RFE_S.iloc[:,0:10]


# In[348]:


####################################################################################
# Select the dependent variables
####################################################################################


# In[14]:


#dependent variable
depVar_OOB = credit_OOB['dpnm']
depVar_OOB_S = credit_OOB_S['dpnm']
depVar_OOB_FS = credit_OOB_FS['dpnm']
depVar_OOB_FS_S = credit_OOB_FS_S['dpnm']
depVar_OOB_RFE = credit_OOB_RFE['dpnm']
depVar_OOB_RFE_S = credit_OOB_RFE_S['dpnm']
depVar_DV = credit_DV['dpnm']
depVar_DV_S = credit_DV_S['dpnm']
depVar_DV_FS = credit_DV_FS['dpnm']
depVar_DV_FS_S = credit_DV_FS_S['dpnm']
depVar_DV_RFE = credit_DV_RFE['dpnm']
depVar_DV_RFE_S = credit_DV_RFE_S['dpnm']


# In[340]:


####################################################################################
# Splitting into train and test sets
####################################################################################


# In[15]:


#Out of box dataset
X_train_OOB, X_test_OOB, y_train_OOB, y_test_OOB = train_test_split(features_OOB, 
                                                                    depVar_OOB, test_size=0.30, random_state = 123)
X_train_OOB.head()


# In[16]:


X_train_OOB.shape, X_test_OOB.shape


# In[17]:


#Out of box scaled dataset
X_train_OOB_S, X_test_OOB_S, y_train_OOB_S, y_test_OOB_S = train_test_split(features_OOB_S, 
                                                                    depVar_OOB_S, test_size=0.30, random_state = 123)
X_train_OOB_S.head()


# In[18]:


X_train_OOB_S.shape, X_test_OOB_S.shape


# In[19]:


#Out of box Feature Selection dataset
X_train_OOB_FS, X_test_OOB_FS, y_train_OOB_FS, y_test_OOB_FS = train_test_split(features_OOB_FS, 
                                                                    depVar_OOB_FS, test_size=0.30, random_state = 123)
X_train_OOB_FS.head()


# In[20]:


X_train_OOB_FS.shape, X_test_OOB_FS.shape


# In[21]:


#Out of box FS scaled dataset
X_train_OOB_FS_S, X_test_OOB_FS_S, y_train_OOB_FS_S, y_test_OOB_FS_S = train_test_split(features_OOB_FS_S, 
                                                                    depVar_OOB_FS_S, test_size=0.30, random_state = 123)
X_train_OOB_FS_S.head()


# In[22]:


X_train_OOB_FS_S.shape, X_test_OOB_FS_S.shape


# In[23]:


#Out of box RFE dataset
X_train_OOB_RFE, X_test_OOB_RFE, y_train_OOB_RFE, y_test_OOB_RFE = train_test_split(features_OOB_RFE, 
                                                                    depVar_OOB_RFE, test_size=0.30, random_state = 123)
X_train_OOB_RFE.head()


# In[24]:


X_train_OOB_RFE.shape, X_test_OOB_RFE.shape


# In[25]:


#Out of box RFE dataset scaled
X_train_OOB_RFE_S, X_test_OOB_RFE_S, y_train_OOB_RFE_S, y_test_OOB_RFE_S = train_test_split(features_OOB_RFE_S, 
                                                                    depVar_OOB_RFE_S, test_size=0.30, random_state = 123)
X_train_OOB_RFE_S.head()


# In[26]:


X_train_OOB_RFE_S.shape, X_test_OOB_RFE_S.shape


# In[27]:


#Dummy Variables w/ binning dataset
X_train_DV, X_test_DV, y_train_DV, y_test_DV = train_test_split(features_DV, 
                                                                    depVar_DV, test_size=0.30, random_state = 123)
X_train_DV.head()


# In[28]:


X_train_DV.shape, X_test_DV.shape


# In[29]:


#Dummy Variables scaled dataset
X_train_DV_S, X_test_DV_S, y_train_DV_S, y_test_DV_S = train_test_split(features_DV_S, 
                                                                    depVar_DV_S, test_size=0.30, random_state = 123)
X_train_DV_S.head()


# In[30]:


X_train_DV_S.shape, X_test_DV_S.shape


# In[31]:


#Dummy Variables Feature Selection dataset
X_train_DV_FS, X_test_DV_FS, y_train_DV_FS, y_test_DV_FS = train_test_split(features_DV_FS, 
                                                                    depVar_DV_FS, test_size=0.30, random_state = 123)
X_train_DV_FS.head()


# In[32]:


X_train_DV_FS.shape, X_test_DV_FS.shape


# In[33]:


#Dummy Variables FS scaled dataset
X_train_DV_FS_S, X_test_DV_FS_S, y_train_DV_FS_S, y_test_DV_FS_S = train_test_split(features_DV_FS_S, 
                                                                    depVar_DV_FS_S, test_size=0.30, random_state = 123)
X_train_DV_FS_S.head()


# In[34]:


X_train_DV_FS_S.shape, X_test_DV_FS_S.shape


# In[35]:


#Dummy Variables RFE
X_train_DV_RFE, X_test_DV_RFE, y_train_DV_RFE, y_test_DV_RFE = train_test_split(features_DV_RFE, 
                                                                    depVar_DV_RFE, test_size=0.30, random_state = 123)
X_train_DV_RFE.head()


# In[36]:


X_train_DV_RFE.shape, X_test_DV_RFE.shape


# In[37]:


#Dummy Variables RFE
X_train_DV_RFE_S, X_test_DV_RFE_S, y_train_DV_RFE_S, y_test_DV_RFE_S = train_test_split(features_DV_RFE_S, 
                                                                    depVar_DV_RFE_S, test_size=0.30, random_state = 123)
X_train_DV_RFE_S.head()


# In[38]:


X_train_DV_RFE_S.shape, X_test_DV_RFE_S.shape


# In[41]:


y_train_OOB_count = len(y_train_OOB.index)
print('The number of observations in the Y training set are:',str(y_train_OOB_count))
y_train_OOB.head(15)


# In[ ]:


####################################################################################
# Model development -- Out of Box Dataframe
####################################################################################


# In[42]:


#Models
modelSVC_OOB = SVC(gamma = 'scale')
modelRF_OOB = RandomForestClassifier(n_estimators = 100)
modelLR_OOB = LinearRegression()
modelLr_OOB = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB = GradientBoostingClassifier()
modelTREE_OOB = tree.DecisionTreeClassifier()
modelKNN_OOB = KNeighborsClassifier(n_neighbors=3)


# In[453]:


#SVC
modelSVC_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelSVC_OOB, X_train_OOB, y_train_OOB)) 
modelSVC_OOB.score(X_train_OOB,y_train_OOB)


# In[454]:


#Random Forest
modelRF_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelRF_OOB, X_train_OOB, y_train_OOB))
modelRF_OOB.score(X_train_OOB,y_train_OOB)


# In[455]:


#Linear Regression
modelLR_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelLR_OOB, X_train_OOB, y_train_OOB))
modelLR_OOB.score(X_train_OOB,y_train_OOB)


# In[44]:


#Logistic Regression
modelLr_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelLr_OOB, X_train_OOB, y_train_OOB, cv=10))
modelLr_OOB.score(X_train_OOB,y_train_OOB)


# In[457]:


#Gradient Boosting
modelGB_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelGB_OOB, X_train_OOB, y_train_OOB))
modelGB_OOB.score(X_train_OOB,y_train_OOB)


# In[458]:


#Decision Tree
modelTREE_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelTREE_OOB, X_train_OOB, y_train_OOB)) 
modelTREE_OOB.score(X_train_OOB,y_train_OOB)


# In[459]:


#KNN
modelKNN_OOB.fit(X_train_OOB,y_train_OOB)
print(cross_val_score(modelKNN_OOB, X_train_OOB, y_train_OOB)) 
modelKNN_OOB.score(X_train_OOB,y_train_OOB)


# In[257]:


####################################################################################
# Evaluating the Results -- Out of Box Dataframe
####################################################################################


# In[480]:


#SVC
predictions_OOB_SVC = modelSVC_OOB.predict(X_test_OOB)
accy_OOB_SVC = accuracy_score(y_test_OOB,predictions_OOB_SVC)
KAPPA_OOB_SVC = cohen_kappa_score(y_test_OOB, predictions_OOB_SVC)
ROC_OOB_SVC = roc_auc_score(y_test_OOB,predictions_OOB_SVC)
prec_OOB_SVC = precision_score(y_test_OOB, predictions_OOB_SVC)
rec_OOB_SVC = recall_score(y_test_OOB,predictions_OOB_SVC)
f1_OOB_SVC = f1_score(y_test_OOB,predictions_OOB_SVC)

model_results =  pd.DataFrame([['SVC', 'OOB', accy_OOB_SVC, KAPPA_OOB_SVC, ROC_OOB_SVC, prec_OOB_SVC, rec_OOB_SVC, f1_OOB_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results


# In[481]:


#Random Forest
predictions_OOB_RF = modelRF_OOB.predict(X_test_OOB)
accy_OOB_RF = accuracy_score(y_test_OOB,predictions_OOB_RF)
KAPPA_OOB_RF = cohen_kappa_score(y_test_OOB, predictions_OOB_RF)
ROC_OOB_RF = roc_auc_score(y_test_OOB,predictions_OOB_RF)
prec_OOB_RF = precision_score(y_test_OOB,predictions_OOB_RF)
rec_OOB_RF = recall_score(y_test_OOB,predictions_OOB_RF)
f1_OOB_RF = f1_score(y_test_OOB,predictions_OOB_RF)

model =  pd.DataFrame([['RF', 'OOB', accy_OOB_RF, KAPPA_OOB_RF, ROC_OOB_RF, prec_OOB_RF, rec_OOB_RF, f1_OOB_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[260]:


#Linear Regression
predictions_OOB_LR = modelLR_OOB.predict(X_test_OOB)
predRsquared_OOB_LR = r2_score(y_test_OOB, predictions_OOB_LR)
rmse_OOB_LR = sqrt(mean_squared_error(y_test_OOB, predictions_OOB_LR))
print('R Squared: %.3f' % predRsquared_OOB_LR)
print('RMSE: %.3f' % rmse_OOB_LR)


# In[482]:


#Logistic Regression
predictions_OOB_Lr = modelLr_OOB.predict(X_test_OOB)
accy_OOB_Lr = accuracy_score(y_test_OOB,predictions_OOB_Lr)
KAPPA_OOB_Lr = cohen_kappa_score(y_test_OOB, predictions_OOB_Lr)
ROC_OOB_Lr = roc_auc_score(y_test_OOB,predictions_OOB_Lr)
prec_OOB_Lr = precision_score(y_test_OOB,predictions_OOB_Lr)
rec_OOB_Lr = recall_score(y_test_OOB,predictions_OOB_Lr)
f1_OOB_Lr = f1_score(y_test_OOB,predictions_OOB_Lr)

model =  pd.DataFrame([['Lr', 'OOB', accy_OOB_Lr, KAPPA_OOB_Lr, ROC_OOB_Lr, prec_OOB_Lr, rec_OOB_Lr, f1_OOB_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[483]:


#Gradient Boosting
predictions_OOB_GB = modelGB_OOB.predict(X_test_OOB)
accy_OOB_GB = accuracy_score(y_test_OOB,predictions_OOB_GB)
KAPPA_OOB_GB = cohen_kappa_score(y_test_OOB, predictions_OOB_GB)
ROC_OOB_GB = roc_auc_score(y_test_OOB,predictions_OOB_GB)
prec_OOB_GB = precision_score(y_test_OOB,predictions_OOB_GB)
rec_OOB_GB = recall_score(y_test_OOB,predictions_OOB_GB)
f1_OOB_GB = f1_score(y_test_OOB,predictions_OOB_GB)

model =  pd.DataFrame([['GB', 'OOB', accy_OOB_GB, KAPPA_OOB_GB, ROC_OOB_GB, prec_OOB_GB, rec_OOB_GB, f1_OOB_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[484]:


#Decision Tree
predictions_OOB_tree = modelTREE_OOB.predict(X_test_OOB)
accy_OOB_Tree = accuracy_score(y_test_OOB,predictions_OOB_tree)
KAPPA_OOB_Tree = cohen_kappa_score(y_test_OOB, predictions_OOB_tree)
ROC_OOB_Tree = roc_auc_score(y_test_OOB, predictions_OOB_tree)
prec_OOB_Tree = precision_score(y_test_OOB,predictions_OOB_tree)
rec_OOB_Tree = recall_score(y_test_OOB,predictions_OOB_tree)
f1_OOB_Tree = f1_score(y_test_OOB,predictions_OOB_tree)

model =  pd.DataFrame([['Tree', 'OOB', accy_OOB_Tree, KAPPA_OOB_Tree, ROC_OOB_Tree, prec_OOB_Tree, rec_OOB_Tree, f1_OOB_Tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[485]:


#KNN
predictions_OOB_KNN = modelKNN_OOB.predict(X_test_OOB)
accy_OOB_KNN = accuracy_score(y_test_OOB,predictions_OOB_KNN)
KAPPA_OOB_KNN = cohen_kappa_score(y_test_OOB, predictions_OOB_KNN)
ROC_OOB_KNN = roc_auc_score(y_test_OOB,predictions_OOB_KNN)
prec_OOB_KNN = precision_score(y_test_OOB,predictions_OOB_KNN)
rec_OOB_KNN = recall_score(y_test_OOB,predictions_OOB_KNN)
f1_OOB_KNN = f1_score(y_test_OOB,predictions_OOB_KNN)

model =  pd.DataFrame([['KNN', 'OOB', accy_OOB_KNN, KAPPA_OOB_KNN, ROC_OOB_KNN, prec_OOB_KNN, rec_OOB_KNN, f1_OOB_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[265]:


outcome_OOB = []
model_names_OOB = []
models_OOB = [('modelSVC_OOB', SVC(gamma = 'scale')),
              ('modelRF_OOB', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB', GradientBoostingClassifier()),
              ('modelTREE_OOB', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB', KNeighborsClassifier(n_neighbors=3))]


# In[266]:


for model_name_OOB, model_OOB in models_OOB:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB = model_selection.cross_val_score(model_OOB, features_OOB, depVar_OOB, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB.append(results_OOB)
    model_names_OOB.append(model_name_OOB)
    output_message_OOB = "%s| Mean=%f STD=%f" % (model_name_OOB, results_OOB.mean(), results_OOB.std())
    print(output_message_OOB)


# In[267]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB)
ax.set_xticklabels(model_names_OOB)
plt.show()


# In[ ]:


### Top Model Selected is modelGB_OOB


# In[ ]:


####################################################################################
# Model development -- Out of Box Scaled
####################################################################################


# In[364]:


#Models
modelSVC_OOB_S = SVC(gamma = 'scale')
modelRF_OOB_S = RandomForestClassifier(n_estimators = 100)
modelLR_OOB_S = LinearRegression()
modelLr_OOB_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB_S = GradientBoostingClassifier()
modelTREE_OOB_S = tree.DecisionTreeClassifier()
modelKNN_OOB_S = KNeighborsClassifier(n_neighbors=3)


# In[365]:


#SVR
modelSVC_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelSVC_OOB_S, X_train_OOB_S, y_train_OOB_S)) 
modelSVC_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[366]:


#Random Forest
modelRF_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelRF_OOB_S, X_train_OOB_S, y_train_OOB_S))
modelRF_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[368]:


#Logistic Regression
modelLr_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelLr_OOB_S, X_train_OOB_S, y_train_OOB_S))
modelLr_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[369]:


#Gradient Boosting
modelGB_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelGB_OOB_S, X_train_OOB_S, y_train_OOB_S))
modelGB_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[370]:


#Decision Tree
modelTREE_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelTREE_OOB_S, X_train_OOB_S, y_train_OOB_S)) 
modelTREE_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[371]:


#KNN
modelKNN_OOB_S.fit(X_train_OOB_S,y_train_OOB_S)
print(cross_val_score(modelKNN_OOB_S, X_train_OOB_S, y_train_OOB_S)) 
modelKNN_OOB_S.score(X_train_OOB_S,y_train_OOB_S)


# In[372]:


####################################################################################
# Evaluating the Results -- Out of Box Scaled Dataframe
####################################################################################


# In[486]:


#SVC
predictions_OOB_S_SVC = modelSVC_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_SVC = accuracy_score(y_test_OOB_S,predictions_OOB_S_SVC)
KAPPA_OOB_S_SVC = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_SVC)
ROC_OOB_S_SVC = roc_auc_score(y_test_OOB,predictions_OOB_S_SVC)
prec_OOB_S_SVC = precision_score(y_test_OOB, predictions_OOB_S_SVC)
rec_OOB_S_SVC = recall_score(y_test_OOB,predictions_OOB_S_SVC)
f1_OOB_S_SVC = f1_score(y_test_OOB,predictions_OOB_S_SVC)

model =  pd.DataFrame([['SVC', 'OOB_S', accy_OOB_S_SVC, KAPPA_OOB_S_SVC, ROC_OOB_S_SVC, prec_OOB_S_SVC, rec_OOB_S_SVC, f1_OOB_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[487]:


#Random Forest
predictions_OOB_S_RF = modelRF_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_RF = accuracy_score(y_test_OOB_S,predictions_OOB_S_RF)
KAPPA_OOB_S_RF = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_RF)
ROC_OOB_S_RF = roc_auc_score(y_test_OOB,predictions_OOB_S_RF)
prec_OOB_S_RF = precision_score(y_test_OOB, predictions_OOB_S_RF)
rec_OOB_S_RF = recall_score(y_test_OOB,predictions_OOB_S_RF)
f1_OOB_S_RF = f1_score(y_test_OOB,predictions_OOB_S_RF)

model =  pd.DataFrame([['RF', 'OOB_S', accy_OOB_S_RF, KAPPA_OOB_S_RF, ROC_OOB_S_RF, prec_OOB_S_RF, rec_OOB_S_RF, f1_OOB_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[488]:


#Logistic Regression
predictions_OOB_S_Lr = modelLr_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_Lr = accuracy_score(y_test_OOB_S,predictions_OOB_S_Lr)
KAPPA_OOB_S_Lr = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_Lr)
ROC_OOB_S_Lr = roc_auc_score(y_test_OOB,predictions_OOB_S_Lr)
prec_OOB_S_Lr = precision_score(y_test_OOB, predictions_OOB_S_Lr)
rec_OOB_S_Lr = recall_score(y_test_OOB,predictions_OOB_S_Lr)
f1_OOB_S_Lr = f1_score(y_test_OOB,predictions_OOB_S_Lr)

model =  pd.DataFrame([['Lr', 'OOB_S', accy_OOB_S_Lr, KAPPA_OOB_S_Lr, ROC_OOB_S_Lr, prec_OOB_S_Lr, rec_OOB_S_Lr, f1_OOB_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[489]:


#Gradient Boosting
predictions_OOB_S_GB = modelGB_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_GB = accuracy_score(y_test_OOB_S,predictions_OOB_S_GB)
KAPPA_OOB_S_GB = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_GB)
ROC_OOB_S_GB = roc_auc_score(y_test_OOB,predictions_OOB_S_GB)
prec_OOB_S_GB = precision_score(y_test_OOB, predictions_OOB_S_GB)
rec_OOB_S_GB = recall_score(y_test_OOB,predictions_OOB_S_GB)
f1_OOB_S_GB = f1_score(y_test_OOB,predictions_OOB_S_GB)

model =  pd.DataFrame([['GB', 'OOB_S', accy_OOB_S_GB, KAPPA_OOB_S_GB, ROC_OOB_S_GB, prec_OOB_S_GB, rec_OOB_S_GB, f1_OOB_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[490]:


#Decision Tree
predictions_OOB_S_tree = modelTREE_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_tree = accuracy_score(y_test_OOB_S,predictions_OOB_S_tree)
KAPPA_OOB_S_tree = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_tree)
ROC_OOB_S_tree = roc_auc_score(y_test_OOB,predictions_OOB_S_tree)
prec_OOB_S_tree = precision_score(y_test_OOB, predictions_OOB_S_tree)
rec_OOB_S_tree = recall_score(y_test_OOB,predictions_OOB_S_tree)
f1_OOB_S_tree = f1_score(y_test_OOB,predictions_OOB_S_tree)

model =  pd.DataFrame([['tree', 'OOB_S', accy_OOB_S_tree, KAPPA_OOB_S_tree, ROC_OOB_S_tree, prec_OOB_S_tree, rec_OOB_S_tree, f1_OOB_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[491]:


#KNN
predictions_OOB_S_KNN = modelKNN_OOB_S.predict(X_test_OOB_S)
accy_OOB_S_KNN = accuracy_score(y_test_OOB_S,predictions_OOB_S_KNN)
KAPPA_OOB_S_KNN = cohen_kappa_score(y_test_OOB_S, predictions_OOB_S_KNN)
ROC_OOB_S_KNN = roc_auc_score(y_test_OOB,predictions_OOB_S_KNN)
prec_OOB_S_KNN = precision_score(y_test_OOB, predictions_OOB_S_KNN)
rec_OOB_S_KNN = recall_score(y_test_OOB,predictions_OOB_S_KNN)
f1_OOB_S_KNN = f1_score(y_test_OOB,predictions_OOB_S_KNN)

model =  pd.DataFrame([['KNN', 'OOB_S', accy_OOB_S_KNN, KAPPA_OOB_S_KNN, ROC_OOB_S_KNN, prec_OOB_S_KNN, rec_OOB_S_KNN, f1_OOB_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True)
model_results


# In[498]:


model_results = model_results.sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[380]:


outcome_OOB_S = []
model_names_OOB_S = []
models_OOB_S = [('modelSVC_OOB_S', SVC(gamma = 'scale')),
              ('modelRF_OOB_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB_S', GradientBoostingClassifier()),
              ('modelTREE_OOB_S', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB_S', KNeighborsClassifier(n_neighbors=3))]


# In[381]:


for model_name_OOB_S, model_OOB_S in models_OOB_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB_S = model_selection.cross_val_score(model_OOB_S, features_OOB_S, depVar_OOB_S, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB_S.append(results_OOB_S)
    model_names_OOB_S.append(model_name_OOB_S)
    output_message_OOB_S = "%s| Mean=%f STD=%f" % (model_name_OOB_S, results_OOB_S.mean(), results_OOB_S.std())
    print(output_message_OOB_S)


# In[382]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB_S)
ax.set_xticklabels(model_names_OOB_S)
plt.show()


# In[287]:


#### Top model selected is modelGB_OOB_S


# In[288]:


####################################################################################
# Model development -- Out of Box Feature Selection
####################################################################################


# In[289]:


#Models
modelSVC_OOB_FS = SVC(gamma = 'scale')
modelRF_OOB_FS = RandomForestClassifier(n_estimators = 100)
modelLR_OOB_FS = LinearRegression()
modelLr_OOB_FS = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB_FS = GradientBoostingClassifier()
modelTREE_OOB_FS = tree.DecisionTreeClassifier()
modelKNN_OOB_FS = KNeighborsClassifier(n_neighbors=3)


# In[290]:


#SVR
modelSVC_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelSVC_OOB_FS, X_train_OOB_FS, y_train_OOB_FS)) 
modelSVC_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[291]:


#Random Forest
modelRF_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelRF_OOB_FS, X_train_OOB_FS, y_train_OOB_FS))
modelRF_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[293]:


#Logistic Regression
modelLr_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelLr_OOB_FS, X_train_OOB_FS, y_train_OOB_FS))
modelLr_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[294]:


#Gradient Boosting
modelGB_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelGB_OOB_FS, X_train_OOB_FS, y_train_OOB_FS))
modelGB_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[295]:


#Decision Tree
modelTREE_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelTREE_OOB_FS, X_train_OOB_FS, y_train_OOB_FS)) 
modelTREE_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[296]:


#KNN
modelKNN_OOB_FS.fit(X_train_OOB_FS,y_train_OOB_FS)
print(cross_val_score(modelKNN_OOB_FS, X_train_OOB_FS, y_train_OOB_FS)) 
modelKNN_OOB_FS.score(X_train_OOB_FS,y_train_OOB_FS)


# In[297]:


####################################################################################
# Evaluating the Results -- Out of Box Feature Selection Dataframe
####################################################################################


# In[499]:


#SVC
predictions_OOB_FS_SVC = modelSVC_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_SVC = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_SVC)
KAPPA_OOB_FS_SVC = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_SVC)
ROC_OOB_FS_SVC = roc_auc_score(y_test_OOB,predictions_OOB_FS_SVC)
prec_OOB_FS_SVC = precision_score(y_test_OOB, predictions_OOB_FS_SVC)
rec_OOB_FS_SVC = recall_score(y_test_OOB,predictions_OOB_FS_SVC)
f1_OOB_FS_SVC = f1_score(y_test_OOB,predictions_OOB_FS_SVC)

model =  pd.DataFrame([['SVC', 'OOB_FS', accy_OOB_FS_SVC, KAPPA_OOB_FS_SVC, ROC_OOB_FS_SVC, prec_OOB_FS_SVC, rec_OOB_FS_SVC, f1_OOB_FS_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[500]:


#Random Forest
predictions_OOB_FS_RF = modelRF_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_RF = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_RF)
KAPPA_OOB_FS_RF = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_RF)
ROC_OOB_FS_RF = roc_auc_score(y_test_OOB,predictions_OOB_FS_RF)
prec_OOB_FS_RF = precision_score(y_test_OOB, predictions_OOB_FS_RF)
rec_OOB_FS_RF = recall_score(y_test_OOB,predictions_OOB_FS_RF)
f1_OOB_FS_RF = f1_score(y_test_OOB,predictions_OOB_FS_RF)

model =  pd.DataFrame([['RF', 'OOB_FS', accy_OOB_FS_RF, KAPPA_OOB_FS_RF, ROC_OOB_FS_RF, prec_OOB_FS_RF, rec_OOB_FS_RF, f1_OOB_FS_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[501]:


#Logistic Regression
predictions_OOB_FS_Lr = modelLr_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_Lr = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_Lr)
KAPPA_OOB_FS_Lr = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_Lr)
ROC_OOB_FS_Lr = roc_auc_score(y_test_OOB,predictions_OOB_FS_Lr)
prec_OOB_FS_Lr = precision_score(y_test_OOB, predictions_OOB_FS_Lr)
rec_OOB_FS_Lr = recall_score(y_test_OOB,predictions_OOB_FS_Lr)
f1_OOB_FS_Lr = f1_score(y_test_OOB,predictions_OOB_FS_Lr)

model =  pd.DataFrame([['Lr', 'OOB_FS', accy_OOB_FS_Lr, KAPPA_OOB_FS_Lr, ROC_OOB_FS_Lr, prec_OOB_FS_Lr, rec_OOB_FS_Lr, f1_OOB_FS_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[502]:


#Gradient Boosting
predictions_OOB_FS_GB = modelGB_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_GB = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_GB)
KAPPA_OOB_FS_GB = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_GB)
ROC_OOB_FS_GB = roc_auc_score(y_test_OOB,predictions_OOB_FS_GB)
prec_OOB_FS_GB = precision_score(y_test_OOB, predictions_OOB_FS_GB)
rec_OOB_FS_GB = recall_score(y_test_OOB,predictions_OOB_FS_GB)
f1_OOB_FS_GB = f1_score(y_test_OOB,predictions_OOB_FS_GB)

model =  pd.DataFrame([['GB', 'OOB_FS', accy_OOB_FS_GB, KAPPA_OOB_FS_GB, ROC_OOB_FS_GB, prec_OOB_FS_GB, rec_OOB_FS_GB, f1_OOB_FS_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[503]:


#Decision Tree
predictions_OOB_FS_tree = modelTREE_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_tree = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_tree)
KAPPA_OOB_FS_tree = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_tree)
ROC_OOB_FS_tree = roc_auc_score(y_test_OOB,predictions_OOB_FS_tree)
prec_OOB_FS_tree = precision_score(y_test_OOB, predictions_OOB_FS_tree)
rec_OOB_FS_tree = recall_score(y_test_OOB,predictions_OOB_FS_tree)
f1_OOB_FS_tree = f1_score(y_test_OOB,predictions_OOB_FS_tree)

model =  pd.DataFrame([['Tree', 'OOB_FS', accy_OOB_FS_tree, KAPPA_OOB_FS_tree, ROC_OOB_FS_tree, prec_OOB_FS_tree, rec_OOB_FS_tree, f1_OOB_FS_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[504]:


#KNN
predictions_OOB_FS_KNN = modelKNN_OOB_FS.predict(X_test_OOB_FS)
accy_OOB_FS_KNN = accuracy_score(y_test_OOB_FS,predictions_OOB_FS_KNN)
KAPPA_OOB_FS_KNN = cohen_kappa_score(y_test_OOB_FS, predictions_OOB_FS_KNN)
ROC_OOB_FS_KNN = roc_auc_score(y_test_OOB,predictions_OOB_FS_KNN)
prec_OOB_FS_KNN = precision_score(y_test_OOB, predictions_OOB_FS_KNN)
rec_OOB_FS_KNN = recall_score(y_test_OOB,predictions_OOB_FS_KNN)
f1_OOB_FS_KNN = f1_score(y_test_OOB,predictions_OOB_FS_KNN)

model =  pd.DataFrame([['KNN', 'OOB_FS', accy_OOB_FS_KNN, KAPPA_OOB_FS_KNN, ROC_OOB_FS_KNN, prec_OOB_FS_KNN, rec_OOB_FS_KNN, f1_OOB_FS_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[305]:


outcome_OOB_FS = []
model_names_OOB_FS = []
models_OOB_FS = [('modelSVC_OOB_FS', SVC(gamma = 'scale')),
              ('modelRF_OOB_FS', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB_FS', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB_FS', GradientBoostingClassifier()),
              ('modelTREE_OOB_FS', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB_FS', KNeighborsClassifier(n_neighbors=3))]


# In[306]:


for model_name_OOB_FS, model_OOB_FS in models_OOB_FS:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB_FS = model_selection.cross_val_score(model_OOB_FS, features_OOB_FS, depVar_OOB_FS, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB_FS.append(results_OOB_FS)
    model_names_OOB_FS.append(model_name_OOB_FS)
    output_message_OOB_FS = "%s| Mean=%f STD=%f" % (model_name_OOB_FS, results_OOB_FS.mean(), results_OOB_FS.std())
    print(output_message_OOB_FS)


# In[307]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB_FS)
ax.set_xticklabels(model_names_OOB_FS)
plt.show()


# In[308]:


#### Top Model selected is modelGB_OOB_FS


# In[309]:


####################################################################################
# Model development -- Out of Box Feature Selection Scaled
####################################################################################


# In[48]:


#Models
modelSVC_OOB_FS_S = SVC(gamma = 'scale')
modelRF_OOB_FS_S = RandomForestClassifier(n_estimators = 100)
modelLR_OOB_FS_S = LinearRegression()
modelLr_OOB_FS_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB_FS_S = GradientBoostingClassifier()
modelTREE_OOB_FS_S = tree.DecisionTreeClassifier()
modelKNN_OOB_FS_S = KNeighborsClassifier(n_neighbors=3)


# In[384]:


#SVR
modelSVC_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelSVC_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S)) 
modelSVC_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[385]:


#Random Forest
modelRF_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelRF_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S))
modelRF_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[387]:


#Logistic Regression
modelLr_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelLr_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S))
modelLr_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[49]:


#Gradient Boosting
modelGB_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelGB_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S))
modelGB_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[389]:


#Decision Tree
modelTREE_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelTREE_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S)) 
modelTREE_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[390]:


#KNN
modelKNN_OOB_FS_S.fit(X_train_OOB_FS_S,y_train_OOB_FS_S)
print(cross_val_score(modelKNN_OOB_FS_S, X_train_OOB_FS_S, y_train_OOB_FS_S)) 
modelKNN_OOB_FS_S.score(X_train_OOB_FS_S,y_train_OOB_FS_S)


# In[391]:


####################################################################################
# Evaluating the Results -- Out of Box Feature Selection Scaled Dataframe
####################################################################################


# In[505]:


#SVC
predictions_OOB_FS_S_SVC = modelSVC_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_SVC = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_SVC)
KAPPA_OOB_FS_S_SVC = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_SVC)
ROC_OOB_FS_S_SVC = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_SVC)
prec_OOB_FS_S_SVC = precision_score(y_test_OOB, predictions_OOB_FS_S_SVC)
rec_OOB_FS_S_SVC = recall_score(y_test_OOB,predictions_OOB_FS_S_SVC)
f1_OOB_FS_S_SVC = f1_score(y_test_OOB,predictions_OOB_FS_S_SVC)

model =  pd.DataFrame([['SVC', 'OOB_FS_S', accy_OOB_FS_S_SVC, KAPPA_OOB_FS_S_SVC, ROC_OOB_FS_S_SVC, prec_OOB_FS_S_SVC, rec_OOB_FS_S_SVC, f1_OOB_FS_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[506]:


#Random Forest
predictions_OOB_FS_S_RF = modelRF_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_RF = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_RF)
KAPPA_OOB_FS_S_RF = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_RF)
ROC_OOB_FS_S_RF = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_RF)
prec_OOB_FS_S_RF = precision_score(y_test_OOB, predictions_OOB_FS_S_RF)
rec_OOB_FS_S_RF = recall_score(y_test_OOB,predictions_OOB_FS_S_RF)
f1_OOB_FS_S_RF = f1_score(y_test_OOB,predictions_OOB_FS_S_RF)

model =  pd.DataFrame([['RF', 'OOB_FS_S', accy_OOB_FS_S_RF, KAPPA_OOB_FS_S_RF, ROC_OOB_FS_S_RF, prec_OOB_FS_S_RF, rec_OOB_FS_S_RF, f1_OOB_FS_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[507]:


#Logistic Regression
predictions_OOB_FS_S_Lr = modelLr_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_Lr = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_Lr)
KAPPA_OOB_FS_S_Lr = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_Lr)
ROC_OOB_FS_S_Lr = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_Lr)
prec_OOB_FS_S_Lr = precision_score(y_test_OOB, predictions_OOB_FS_S_Lr)
rec_OOB_FS_S_Lr = recall_score(y_test_OOB,predictions_OOB_FS_S_Lr)
f1_OOB_FS_S_Lr = f1_score(y_test_OOB,predictions_OOB_FS_S_Lr)

model =  pd.DataFrame([['Lr', 'OOB_FS_S', accy_OOB_FS_S_Lr, KAPPA_OOB_FS_S_Lr, ROC_OOB_FS_S_Lr, prec_OOB_FS_S_Lr, rec_OOB_FS_S_Lr, f1_OOB_FS_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[508]:


#Gradient Boosting
predictions_OOB_FS_S_GB = modelGB_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_GB = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_GB)
KAPPA_OOB_FS_S_GB = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_GB)
ROC_OOB_FS_S_GB = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_GB)
prec_OOB_FS_S_GB = precision_score(y_test_OOB, predictions_OOB_FS_S_GB)
rec_OOB_FS_S_GB = recall_score(y_test_OOB,predictions_OOB_FS_S_GB)
f1_OOB_FS_S_GB = f1_score(y_test_OOB,predictions_OOB_FS_S_GB)

model =  pd.DataFrame([['GB', 'OOB_FS_S', accy_OOB_FS_S_GB, KAPPA_OOB_FS_S_GB, ROC_OOB_FS_S_GB, prec_OOB_FS_S_GB, rec_OOB_FS_S_GB, f1_OOB_FS_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[509]:


#Decision Tree
predictions_OOB_FS_S_tree = modelTREE_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_tree = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_tree)
KAPPA_OOB_FS_S_tree = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_tree)
ROC_OOB_FS_S_tree = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_tree)
prec_OOB_FS_S_tree = precision_score(y_test_OOB, predictions_OOB_FS_S_tree)
rec_OOB_FS_S_tree = recall_score(y_test_OOB,predictions_OOB_FS_S_tree)
f1_OOB_FS_S_tree = f1_score(y_test_OOB,predictions_OOB_FS_S_tree)

model =  pd.DataFrame([['Tree', 'OOB_FS_S', accy_OOB_FS_S_tree, KAPPA_OOB_FS_S_tree, ROC_OOB_FS_S_tree, prec_OOB_FS_S_tree, rec_OOB_FS_S_tree, f1_OOB_FS_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[510]:


#KNN
predictions_OOB_FS_S_KNN = modelKNN_OOB_FS_S.predict(X_test_OOB_FS_S)
accy_OOB_FS_S_KNN = accuracy_score(y_test_OOB_FS_S,predictions_OOB_FS_S_KNN)
KAPPA_OOB_FS_S_KNN = cohen_kappa_score(y_test_OOB_FS_S, predictions_OOB_FS_S_KNN)
ROC_OOB_FS_S_KNN = roc_auc_score(y_test_OOB,predictions_OOB_FS_S_KNN)
prec_OOB_FS_S_KNN = precision_score(y_test_OOB, predictions_OOB_FS_S_KNN)
rec_OOB_FS_S_KNN = recall_score(y_test_OOB,predictions_OOB_FS_S_KNN)
f1_OOB_FS_S_KNN = f1_score(y_test_OOB,predictions_OOB_FS_S_KNN)

model =  pd.DataFrame([['KNN', 'OOB_FS_S', accy_OOB_FS_S_KNN, KAPPA_OOB_FS_S_KNN, ROC_OOB_FS_S_KNN, prec_OOB_FS_S_KNN, rec_OOB_FS_S_KNN, f1_OOB_FS_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[399]:


outcome_OOB_FS_S = []
model_names_OOB_FS_S = []
models_OOB_FS_S = [('modelSVC_OOB_FS_S', SVC(gamma = 'scale')),
              ('modelRF_OOB_FS_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB_FS_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB_FS_S', GradientBoostingClassifier()),
              ('modelTREE_OOB_FS_S', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB_FS_S', KNeighborsClassifier(n_neighbors=3))]


# In[400]:


for model_name_OOB_FS_S, model_OOB_FS_S in models_OOB_FS_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB_FS_S = model_selection.cross_val_score(model_OOB_FS_S, features_OOB_FS_S, depVar_OOB_FS_S, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB_FS_S.append(results_OOB_FS_S)
    model_names_OOB_FS_S.append(model_name_OOB_FS_S)
    output_message_OOB_FS_S = "%s| Mean=%f STD=%f" % (model_name_OOB_FS_S, results_OOB_FS_S.mean(), results_OOB_FS_S.std())
    print(output_message_OOB_FS_S)


# In[401]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB_FS_S)
ax.set_xticklabels(model_names_OOB_FS_S)
plt.show()


# In[ ]:


#### Top model selected is modelGBOOB_FS_S


# In[ ]:


####################################################################################
# Model development -- Dummy Variable Dataframe
####################################################################################


# In[402]:


#Models
modelSVC_DV = SVC(gamma = 'scale')
modelRF_DV = RandomForestClassifier(n_estimators = 100)
modelLR_DV = LinearRegression()
modelLr_DV = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV = GradientBoostingClassifier()
modelTREE_DV = tree.DecisionTreeClassifier()
modelKNN_DV = KNeighborsClassifier(n_neighbors=3)


# In[403]:


#SVC
modelSVC_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelSVC_DV, X_train_DV, y_train_DV)) 
modelSVC_DV.score(X_train_DV,y_train_DV)


# In[404]:


#Random Forest
modelRF_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelRF_DV, X_train_DV, y_train_DV))
modelRF_DV.score(X_train_DV,y_train_DV)


# In[406]:


#Logistic Regression
modelLr_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelLr_DV, X_train_DV, y_train_DV))
modelLr_DV.score(X_train_DV,y_train_DV)


# In[407]:


#Gradient Boosting
modelGB_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelGB_DV, X_train_DV, y_train_DV))
modelGB_DV.score(X_train_DV,y_train_DV)


# In[408]:


#Decision Tree
modelTREE_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelTREE_DV, X_train_DV, y_train_DV)) 
modelTREE_DV.score(X_train_DV,y_train_DV)


# In[409]:


#KNN
modelKNN_DV.fit(X_train_DV,y_train_DV)
print(cross_val_score(modelKNN_DV, X_train_DV, y_train_DV)) 
modelKNN_DV.score(X_train_DV,y_train_DV)


# In[410]:


####################################################################################
# Evaluating the Results -- Dummy Variables Dataframe
####################################################################################


# In[511]:


#SVC
predictions_DV_SVC = modelSVC_DV.predict(X_test_DV)
accy_DV_SVC = accuracy_score(y_test_DV,predictions_DV_SVC)
KAPPA_DV_SVC = cohen_kappa_score(y_test_DV, predictions_DV_SVC)
ROC_DV_SVC = roc_auc_score(y_test_OOB,predictions_DV_SVC)
prec_DV_SVC = precision_score(y_test_OOB, predictions_DV_SVC)
rec_DV_SVC = recall_score(y_test_OOB,predictions_DV_SVC)
f1_DV_SVC = f1_score(y_test_OOB,predictions_DV_SVC)

model =  pd.DataFrame([['SVC', 'DV', accy_DV_SVC, KAPPA_DV_SVC, ROC_DV_SVC, prec_DV_SVC, rec_DV_SVC, f1_DV_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[512]:


#Random Forest
predictions_DV_RF = modelRF_DV.predict(X_test_DV)
accy_DV_RF = accuracy_score(y_test_DV,predictions_DV_RF)
KAPPA_DV_RF = cohen_kappa_score(y_test_DV, predictions_DV_RF)
ROC_DV_RF = roc_auc_score(y_test_OOB,predictions_DV_RF)
prec_DV_RF = precision_score(y_test_OOB, predictions_DV_RF)
rec_DV_RF = recall_score(y_test_OOB,predictions_DV_RF)
f1_DV_RF = f1_score(y_test_OOB,predictions_DV_RF)

model =  pd.DataFrame([['RF', 'DV', accy_DV_RF, KAPPA_DV_RF, ROC_DV_RF, prec_DV_RF, rec_DV_RF, f1_DV_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[513]:


#Logistic Regression
predictions_DV_Lr = modelLr_DV.predict(X_test_DV)
accy_DV_Lr = accuracy_score(y_test_DV,predictions_DV_Lr)
KAPPA_DV_Lr = cohen_kappa_score(y_test_DV, predictions_DV_Lr)
ROC_DV_Lr = roc_auc_score(y_test_OOB,predictions_DV_Lr)
prec_DV_Lr = precision_score(y_test_OOB, predictions_DV_Lr)
rec_DV_Lr = recall_score(y_test_OOB,predictions_DV_Lr)
f1_DV_Lr = f1_score(y_test_OOB,predictions_DV_Lr)

model =  pd.DataFrame([['Lr', 'DV', accy_DV_Lr, KAPPA_DV_Lr, ROC_DV_Lr, prec_DV_Lr, rec_DV_Lr, f1_DV_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[514]:


#Gradient Boosting
predictions_DV_GB = modelGB_DV.predict(X_test_DV)
accy_DV_GB = accuracy_score(y_test_DV,predictions_DV_GB)
KAPPA_DV_GB = cohen_kappa_score(y_test_DV, predictions_DV_GB)
ROC_DV_GB = roc_auc_score(y_test_OOB,predictions_DV_GB)
prec_DV_GB = precision_score(y_test_OOB, predictions_DV_GB)
rec_DV_GB = recall_score(y_test_OOB,predictions_DV_GB)
f1_DV_GB = f1_score(y_test_OOB,predictions_DV_GB)

model =  pd.DataFrame([['GB', 'DV', accy_DV_GB, KAPPA_DV_GB, ROC_DV_GB, prec_DV_GB, rec_DV_GB, f1_DV_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[515]:


#Decision Tree
predictions_DV_tree = modelTREE_DV.predict(X_test_DV)
accy_DV_tree = accuracy_score(y_test_DV,predictions_DV_tree)
KAPPA_DV_tree = cohen_kappa_score(y_test_DV, predictions_DV_tree)
ROC_DV_tree = roc_auc_score(y_test_OOB,predictions_DV_tree)
prec_DV_tree = precision_score(y_test_OOB, predictions_DV_tree)
rec_DV_tree = recall_score(y_test_OOB,predictions_DV_tree)
f1_DV_tree = f1_score(y_test_OOB,predictions_DV_tree)

model =  pd.DataFrame([['Tree', 'DV', accy_DV_tree, KAPPA_DV_tree, ROC_DV_tree, prec_DV_tree, rec_DV_tree, f1_DV_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[516]:


#KNN
predictions_DV_KNN = modelKNN_DV.predict(X_test_DV)
accy_DV_KNN = accuracy_score(y_test_DV,predictions_DV_KNN)
KAPPA_DV_KNN = cohen_kappa_score(y_test_DV, predictions_DV_KNN)
ROC_DV_KNN = roc_auc_score(y_test_OOB,predictions_DV_KNN)
prec_DV_KNN = precision_score(y_test_OOB, predictions_DV_KNN)
rec_DV_KNN = recall_score(y_test_OOB,predictions_DV_KNN)
f1_DV_KNN = f1_score(y_test_OOB,predictions_DV_KNN)

model =  pd.DataFrame([['KNN', 'DV', accy_DV_KNN, KAPPA_DV_KNN, ROC_DV_KNN, prec_DV_KNN, rec_DV_KNN, f1_DV_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[418]:


outcome_DV = []
model_names_DV = []
models_DV = [('modelSVC_DV', SVC(gamma = 'scale')),
              ('modelRF_DV', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV', GradientBoostingClassifier()),
              ('modelTREE_DV', tree.DecisionTreeClassifier()),
              ('modelKNN_DV', KNeighborsClassifier(n_neighbors=3))]


# In[419]:


for model_name_DV, model_DV in models_DV:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV = model_selection.cross_val_score(model_DV, features_DV, depVar_DV, cv=k_fold_validation, scoring='accuracy')
    outcome_DV.append(results_DV)
    model_names_DV.append(model_name_DV)
    output_message_DV = "%s| Mean=%f STD=%f" % (model_name_DV, results_DV.mean(), results_DV.std())
    print(output_message_DV)


# In[420]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV)
ax.set_xticklabels(model_names_DV)
plt.show()


# In[ ]:


#### Top model is modelGB_DV


# In[ ]:


####################################################################################
# Model development -- Dummy Variables Scaled
####################################################################################


# In[421]:


#Models
modelSVC_DV_S = SVC(gamma = 'scale')
modelRF_DV_S = RandomForestClassifier(n_estimators = 100)
modelLR_DV_S = LinearRegression()
modelLr_DV_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV_S = GradientBoostingClassifier()
modelTREE_DV_S = tree.DecisionTreeClassifier()
modelKNN_DV_S = KNeighborsClassifier(n_neighbors=3)


# In[422]:


#SVR
modelSVC_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelSVC_DV_S, X_train_DV_S, y_train_DV_S)) 
modelSVC_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[423]:


#Random Forest
modelRF_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelRF_DV_S, X_train_DV_S, y_train_DV_S))
modelRF_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[424]:


#Linear Regression
modelLR_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelLR_DV_S, X_train_DV_S, y_train_DV_S))
modelLR_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[425]:


#Logistic Regression
modelLr_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelLr_DV_S, X_train_DV_S, y_train_DV_S))
modelLr_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[426]:


#Gradient Boosting
modelGB_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelGB_DV_S, X_train_DV_S, y_train_DV_S))
modelGB_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[427]:


#Decision Tree
modelTREE_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelTREE_DV_S, X_train_DV_S, y_train_DV_S)) 
modelTREE_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[428]:


#KNN
modelKNN_DV_S.fit(X_train_DV_S,y_train_DV_S)
print(cross_val_score(modelKNN_DV_S, X_train_DV_S, y_train_DV_S)) 
modelKNN_DV_S.score(X_train_DV_S,y_train_DV_S)


# In[429]:


####################################################################################
# Evaluating the Results -- Dummy Variables Scaled Dataframe
####################################################################################


# In[520]:


#SVC
predictions_DV_S_SVC = modelSVC_DV_S.predict(X_test_DV_S)
accy_DV_S_SVC = accuracy_score(y_test_DV_S,predictions_DV_S_SVC)
KAPPA_DV_S_SVC = cohen_kappa_score(y_test_DV_S, predictions_DV_S_SVC)
ROC_DV_S_SVC = roc_auc_score(y_test_OOB,predictions_DV_S_SVC)
prec_DV_S_SVC = precision_score(y_test_OOB, predictions_DV_S_SVC)
rec_DV_S_SVC = recall_score(y_test_OOB,predictions_DV_S_SVC)
f1_DV_S_SVC = f1_score(y_test_OOB,predictions_DV_S_SVC)

model =  pd.DataFrame([['SVC', 'DV_S', accy_DV_S_SVC, KAPPA_DV_S_SVC, ROC_DV_S_SVC, prec_DV_S_SVC, rec_DV_S_SVC, f1_DV_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[521]:


#Random Forest
predictions_DV_S_RF = modelRF_DV_S.predict(X_test_DV_S)
accy_DV_S_RF = accuracy_score(y_test_DV_S,predictions_DV_S_RF)
KAPPA_DV_S_RF = cohen_kappa_score(y_test_DV_S, predictions_DV_S_RF)
ROC_DV_S_RF = roc_auc_score(y_test_OOB,predictions_DV_S_RF)
prec_DV_S_RF = precision_score(y_test_OOB, predictions_DV_S_RF)
rec_DV_S_RF = recall_score(y_test_OOB,predictions_DV_S_RF)
f1_DV_S_RF = f1_score(y_test_OOB,predictions_DV_S_RF)

model =  pd.DataFrame([['RF', 'DV_S', accy_DV_S_RF, KAPPA_DV_S_RF, ROC_DV_S_RF, prec_DV_S_RF, rec_DV_S_RF, f1_DV_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[522]:


#Logistic Regression
predictions_DV_S_Lr = modelLr_DV_S.predict(X_test_DV_S)
accy_DV_S_Lr = accuracy_score(y_test_DV_S,predictions_DV_S_Lr)
KAPPA_DV_S_Lr = cohen_kappa_score(y_test_DV_S, predictions_DV_S_Lr)
ROC_DV_S_Lr = roc_auc_score(y_test_OOB,predictions_DV_S_Lr)
prec_DV_S_Lr = precision_score(y_test_OOB, predictions_DV_S_Lr)
rec_DV_S_Lr = recall_score(y_test_OOB,predictions_DV_S_Lr)
f1_DV_S_Lr = f1_score(y_test_OOB,predictions_DV_S_Lr)

model =  pd.DataFrame([['Lr', 'DV_S', accy_DV_S_Lr, KAPPA_DV_S_Lr, ROC_DV_S_Lr, prec_DV_S_Lr, rec_DV_S_Lr, f1_DV_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[523]:


# Gradient Boosting
predictions_DV_S_GB = modelGB_DV_S.predict(X_test_DV_S)
accy_DV_S_GB = accuracy_score(y_test_DV_S,predictions_DV_S_GB)
KAPPA_DV_S_GB = cohen_kappa_score(y_test_DV_S, predictions_DV_S_GB)
ROC_DV_S_GB = roc_auc_score(y_test_OOB,predictions_DV_S_GB)
prec_DV_S_GB = precision_score(y_test_OOB, predictions_DV_S_GB)
rec_DV_S_GB = recall_score(y_test_OOB,predictions_DV_S_GB)
f1_DV_S_GB = f1_score(y_test_OOB,predictions_DV_S_GB)

model =  pd.DataFrame([['GB', 'DV_S', accy_DV_S_GB, KAPPA_DV_S_GB, ROC_DV_S_GB, prec_DV_S_GB, rec_DV_S_GB, f1_DV_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[524]:


# Decision Tree
predictions_DV_S_tree = modelTREE_DV_S.predict(X_test_DV_S)
accy_DV_S_tree = accuracy_score(y_test_DV_S,predictions_DV_S_tree)
KAPPA_DV_S_tree = cohen_kappa_score(y_test_DV_S, predictions_DV_S_tree)
ROC_DV_S_tree = roc_auc_score(y_test_OOB,predictions_DV_S_tree)
prec_DV_S_tree = precision_score(y_test_OOB, predictions_DV_S_tree)
rec_DV_S_tree = recall_score(y_test_OOB,predictions_DV_S_tree)
f1_DV_S_tree = f1_score(y_test_OOB,predictions_DV_S_tree)

model =  pd.DataFrame([['Tree', 'DV_S', accy_DV_S_tree, KAPPA_DV_S_tree, ROC_DV_S_tree, prec_DV_S_tree, rec_DV_S_tree, f1_DV_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[525]:


#KNN
predictions_DV_S_KNN = modelKNN_DV_S.predict(X_test_DV_S)
accy_DV_S_KNN = accuracy_score(y_test_DV_S,predictions_DV_S_KNN)
KAPPA_DV_S_KNN = cohen_kappa_score(y_test_DV_S, predictions_DV_S_KNN)
ROC_DV_S_KNN = roc_auc_score(y_test_OOB,predictions_DV_S_KNN)
prec_DV_S_KNN = precision_score(y_test_OOB, predictions_DV_S_KNN)
rec_DV_S_KNN = recall_score(y_test_OOB,predictions_DV_S_KNN)
f1_DV_S_KNN = f1_score(y_test_OOB,predictions_DV_S_KNN)

model =  pd.DataFrame([['KNN', 'DV_S', accy_DV_S_KNN, KAPPA_DV_S_KNN, ROC_DV_S_KNN, prec_DV_S_KNN, rec_DV_S_KNN, f1_DV_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[450]:


outcome_DV_S = []
model_names_DV_S = []
models_DV_S = [('modelSVC_DV_S', SVC(gamma = 'scale')),
              ('modelRF_DV_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV_S', GradientBoostingClassifier()),
              ('modelTREE_DV_S', tree.DecisionTreeClassifier()),
              ('modelKNN_DV_S', KNeighborsClassifier(n_neighbors=3))]


# In[451]:


for model_name_DV_S, model_DV_S in models_DV_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV_S = model_selection.cross_val_score(model_DV_S, features_DV_S, depVar_DV_S, cv=k_fold_validation, scoring='accuracy')
    outcome_DV_S.append(results_DV_S)
    model_names_DV_S.append(model_name_DV_S)
    output_message_DV_S = "%s| Mean=%f STD=%f" % (model_name_DV_S, results_DV_S.mean(), results_DV_S.std())
    print(output_message_DV_S)


# In[526]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV_S)
ax.set_xticklabels(model_names_DV_S)
plt.show()


# In[ ]:


##### Top model is modelSVC_DV_S


# In[ ]:


####################################################################################
# Model development -- Dummy Variables Feature Selection
####################################################################################


# In[527]:


#Models
modelSVC_DV_FS = SVC(gamma = 'scale')
modelRF_DV_FS = RandomForestClassifier(n_estimators = 100)
modelLR_DV_FS = LinearRegression()
modelLr_DV_FS = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV_FS = GradientBoostingClassifier()
modelTREE_DV_FS = tree.DecisionTreeClassifier()
modelKNN_DV_FS = KNeighborsClassifier(n_neighbors=3)


# In[528]:


#SVR
modelSVC_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelSVC_DV_FS, X_train_DV_FS, y_train_DV_FS)) 
modelSVC_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[529]:


#Random Forest
modelRF_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelRF_DV_FS, X_train_DV_FS, y_train_DV_FS))
modelRF_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[530]:


#Linear Regression
modelLR_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelLR_DV_FS, X_train_DV_FS, y_train_DV_FS))
modelLR_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[531]:


#Logistic Regression
modelLr_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelLr_DV_FS, X_train_DV_FS, y_train_DV_FS))
modelLr_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[532]:


#Gradient Boosting
modelGB_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelGB_DV_FS, X_train_DV_FS, y_train_DV_FS))
modelGB_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[533]:


#Decision Tree
modelTREE_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelTREE_DV_FS, X_train_DV_FS, y_train_DV_FS)) 
modelTREE_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[534]:


#KNN
modelKNN_DV_FS.fit(X_train_DV_FS,y_train_DV_FS)
print(cross_val_score(modelKNN_DV_FS, X_train_DV_FS, y_train_DV_FS)) 
modelKNN_DV_FS.score(X_train_DV_FS,y_train_DV_FS)


# In[535]:


####################################################################################
# Evaluating the Results -- Dummy Variables Feature Selection Dataframe
####################################################################################


# In[538]:


#SVC
predictions_DV_FS_SVC = modelSVC_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_SVC = accuracy_score(y_test_DV_FS,predictions_DV_FS_SVC)
KAPPA_DV_FS_SVC = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_SVC)
ROC_DV_FS_SVC = roc_auc_score(y_test_OOB,predictions_DV_FS_SVC)
prec_DV_FS_SVC = precision_score(y_test_OOB, predictions_DV_FS_SVC)
rec_DV_FS_SVC = recall_score(y_test_OOB,predictions_DV_FS_SVC)
f1_DV_FS_SVC = f1_score(y_test_OOB,predictions_DV_FS_SVC)

model =  pd.DataFrame([['SVC', 'DV_FS', accy_DV_FS_SVC, KAPPA_DV_FS_SVC, ROC_DV_FS_SVC, prec_DV_FS_SVC, rec_DV_FS_SVC, f1_DV_FS_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[539]:


#Random Forest
predictions_DV_FS_RF = modelRF_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_RF = accuracy_score(y_test_DV_FS,predictions_DV_FS_RF)
KAPPA_DV_FS_RF = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_RF)
ROC_DV_FS_RF = roc_auc_score(y_test_OOB,predictions_DV_FS_RF)
prec_DV_FS_RF = precision_score(y_test_OOB, predictions_DV_FS_RF)
rec_DV_FS_RF = recall_score(y_test_OOB,predictions_DV_FS_RF)
f1_DV_FS_RF = f1_score(y_test_OOB,predictions_DV_FS_RF)

model =  pd.DataFrame([['RF', 'DV_FS', accy_DV_FS_RF, KAPPA_DV_FS_RF, ROC_DV_FS_RF, prec_DV_FS_RF, rec_DV_FS_RF, f1_DV_FS_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[540]:


#Logistic Regression
predictions_DV_FS_Lr = modelLr_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_Lr = accuracy_score(y_test_DV_FS,predictions_DV_FS_Lr)
KAPPA_DV_FS_Lr = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_Lr)
ROC_DV_FS_Lr = roc_auc_score(y_test_OOB,predictions_DV_FS_Lr)
prec_DV_FS_Lr = precision_score(y_test_OOB, predictions_DV_FS_Lr)
rec_DV_FS_Lr = recall_score(y_test_OOB,predictions_DV_FS_Lr)
f1_DV_FS_Lr = f1_score(y_test_OOB,predictions_DV_FS_Lr)

model =  pd.DataFrame([['Lr', 'DV_FS', accy_DV_FS_Lr, KAPPA_DV_FS_Lr, ROC_DV_FS_Lr, prec_DV_FS_Lr, rec_DV_FS_Lr, f1_DV_FS_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[541]:


#Gradient Boosting
predictions_DV_FS_GB = modelGB_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_GB = accuracy_score(y_test_DV_FS,predictions_DV_FS_GB)
KAPPA_DV_FS_GB = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_GB)
ROC_DV_FS_GB = roc_auc_score(y_test_OOB,predictions_DV_FS_GB)
prec_DV_FS_GB = precision_score(y_test_OOB, predictions_DV_FS_GB)
rec_DV_FS_GB = recall_score(y_test_OOB,predictions_DV_FS_GB)
f1_DV_FS_GB = f1_score(y_test_OOB,predictions_DV_FS_GB)

model =  pd.DataFrame([['GB', 'DV_FS', accy_DV_FS_GB, KAPPA_DV_FS_GB, ROC_DV_FS_GB, prec_DV_FS_GB, rec_DV_FS_GB, f1_DV_FS_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[542]:


#Decision Tree
predictions_DV_FS_tree = modelTREE_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_tree = accuracy_score(y_test_DV_FS,predictions_DV_FS_tree)
KAPPA_DV_FS_tree = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_tree)
ROC_DV_FS_tree = roc_auc_score(y_test_OOB,predictions_DV_FS_tree)
prec_DV_FS_tree = precision_score(y_test_OOB, predictions_DV_FS_tree)
rec_DV_FS_tree = recall_score(y_test_OOB,predictions_DV_FS_tree)
f1_DV_FS_tree = f1_score(y_test_OOB,predictions_DV_FS_tree)

model =  pd.DataFrame([['Tree', 'DV_FS', accy_DV_FS_tree, KAPPA_DV_FS_tree, ROC_DV_FS_tree, prec_DV_FS_tree, rec_DV_FS_tree, f1_DV_FS_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[543]:


#KNN
predictions_DV_FS_KNN = modelKNN_DV_FS.predict(X_test_DV_FS)
accy_DV_FS_KNN = accuracy_score(y_test_DV_FS,predictions_DV_FS_KNN)
KAPPA_DV_FS_KNN = cohen_kappa_score(y_test_DV_FS, predictions_DV_FS_KNN)
ROC_DV_FS_KNN = roc_auc_score(y_test_OOB,predictions_DV_FS_KNN)
prec_DV_FS_KNN = precision_score(y_test_OOB, predictions_DV_FS_KNN)
rec_DV_FS_KNN = recall_score(y_test_OOB,predictions_DV_FS_KNN)
f1_DV_FS_KNN = f1_score(y_test_OOB,predictions_DV_FS_KNN)

model =  pd.DataFrame([['KNN', 'DV_FS', accy_DV_FS_KNN, KAPPA_DV_FS_KNN, ROC_DV_FS_KNN, prec_DV_FS_KNN, rec_DV_FS_KNN, f1_DV_FS_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[544]:


outcome_DV_FS = []
model_names_DV_FS = []
models_DV_FS = [('modelSVC_DV_FS', SVC(gamma = 'scale')),
              ('modelRF_DV_FS', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV_FS', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV_FS', GradientBoostingClassifier()),
              ('modelTREE_DV_FS', tree.DecisionTreeClassifier()),
              ('modelKNN_DV_FS', KNeighborsClassifier(n_neighbors=3))]


# In[545]:


for model_name_DV_FS, model_DV_FS in models_DV_FS:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV_FS = model_selection.cross_val_score(model_DV_FS, features_DV_FS, depVar_DV_FS, cv=k_fold_validation, scoring='accuracy')
    outcome_DV_FS.append(results_DV_FS)
    model_names_DV_FS.append(model_name_DV_FS)
    output_message_DV_FS = "%s| Mean=%f STD=%f" % (model_name_DV_FS, results_DV_FS.mean(), results_DV_FS.std())
    print(output_message_DV_FS)


# In[546]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV_FS)
ax.set_xticklabels(model_names_DV_FS)
plt.show()


# In[ ]:


#### Top model for DV_FS is modelGB_DV_FS


# In[ ]:


####################################################################################
# Model development -- Dummy Variables Feature Selection Scaled
####################################################################################


# In[547]:


#Models
modelSVC_DV_FS_S = SVC(gamma = 'scale')
modelRF_DV_FS_S = RandomForestClassifier(n_estimators = 100)
modelLR_DV_FS_S = LinearRegression()
modelLr_DV_FS_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV_FS_S = GradientBoostingClassifier()
modelTREE_DV_FS_S = tree.DecisionTreeClassifier()
modelKNN_DV_FS_S = KNeighborsClassifier(n_neighbors=3)


# In[548]:


#SVR
modelSVC_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelSVC_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S)) 
modelSVC_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[549]:


#Random Forest
modelRF_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelRF_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S))
modelRF_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[550]:


#Linear Regression
modelLR_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelLR_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S))
modelLR_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[551]:


#Logistic Regression
modelLr_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelLr_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S))
modelLr_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[552]:


#Gradient Boosting
modelGB_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelGB_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S))
modelGB_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[553]:


#Decision Tree
modelTREE_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelTREE_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S)) 
modelTREE_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[554]:


#KNN
modelKNN_DV_FS_S.fit(X_train_DV_FS_S,y_train_DV_FS_S)
print(cross_val_score(modelKNN_DV_FS_S, X_train_DV_FS_S, y_train_DV_FS_S)) 
modelKNN_DV_FS_S.score(X_train_DV_FS_S,y_train_DV_FS_S)


# In[555]:


####################################################################################
# Evaluating the Results -- Dummy Variables Feature Selection Scaled Dataframe
####################################################################################


# In[556]:


#SVC
predictions_DV_FS_S_SVC = modelSVC_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_SVC = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_SVC)
KAPPA_DV_FS_S_SVC = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_SVC)
ROC_DV_FS_S_SVC = roc_auc_score(y_test_OOB,predictions_DV_FS_S_SVC)
prec_DV_FS_S_SVC = precision_score(y_test_OOB, predictions_DV_FS_S_SVC)
rec_DV_FS_S_SVC = recall_score(y_test_OOB,predictions_DV_FS_S_SVC)
f1_DV_FS_S_SVC = f1_score(y_test_OOB,predictions_DV_FS_S_SVC)

model =  pd.DataFrame([['SVC', 'DV_S_FS', accy_DV_FS_S_SVC, KAPPA_DV_FS_S_SVC, ROC_DV_FS_S_SVC, prec_DV_FS_S_SVC, rec_DV_FS_S_SVC, f1_DV_FS_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[557]:


#Random Forest
predictions_DV_FS_S_RF = modelRF_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_RF = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_RF)
KAPPA_DV_FS_S_RF = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_RF)
ROC_DV_FS_S_RF = roc_auc_score(y_test_OOB,predictions_DV_FS_S_RF)
prec_DV_FS_S_RF = precision_score(y_test_OOB, predictions_DV_FS_S_RF)
rec_DV_FS_S_RF = recall_score(y_test_OOB,predictions_DV_FS_S_RF)
f1_DV_FS_S_RF = f1_score(y_test_OOB,predictions_DV_FS_S_RF)

model =  pd.DataFrame([['RF', 'DV_S_FS', accy_DV_FS_S_RF, KAPPA_DV_FS_S_RF, ROC_DV_FS_S_RF, prec_DV_FS_S_RF, rec_DV_FS_S_RF, f1_DV_FS_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[558]:


#Logistic Regression
predictions_DV_FS_S_Lr = modelLr_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_Lr = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_Lr)
KAPPA_DV_FS_S_Lr = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_Lr)
ROC_DV_FS_S_Lr = roc_auc_score(y_test_OOB,predictions_DV_FS_S_Lr)
prec_DV_FS_S_Lr = precision_score(y_test_OOB, predictions_DV_FS_S_Lr)
rec_DV_FS_S_Lr = recall_score(y_test_OOB,predictions_DV_FS_S_Lr)
f1_DV_FS_S_Lr = f1_score(y_test_OOB,predictions_DV_FS_S_Lr)

model =  pd.DataFrame([['Lr', 'DV_S_FS', accy_DV_FS_S_Lr, KAPPA_DV_FS_S_Lr, ROC_DV_FS_S_Lr, prec_DV_FS_S_Lr, rec_DV_FS_S_Lr, f1_DV_FS_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[559]:


#Gradient Boosting
predictions_DV_FS_S_GB = modelGB_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_GB = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_GB)
KAPPA_DV_FS_S_GB = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_GB)
ROC_DV_FS_S_GB = roc_auc_score(y_test_OOB,predictions_DV_FS_S_GB)
prec_DV_FS_S_GB = precision_score(y_test_OOB, predictions_DV_FS_S_GB)
rec_DV_FS_S_GB = recall_score(y_test_OOB,predictions_DV_FS_S_GB)
f1_DV_FS_S_GB = f1_score(y_test_OOB,predictions_DV_FS_S_GB)

model =  pd.DataFrame([['GB', 'DV_S_FS', accy_DV_FS_S_GB, KAPPA_DV_FS_S_GB, ROC_DV_FS_S_GB, prec_DV_FS_S_GB, rec_DV_FS_S_GB, f1_DV_FS_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[560]:


#Decision Tree
predictions_DV_FS_S_tree = modelTREE_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_tree = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_tree)
KAPPA_DV_FS_S_tree = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_tree)
ROC_DV_FS_S_tree = roc_auc_score(y_test_OOB,predictions_DV_FS_S_tree)
prec_DV_FS_S_tree = precision_score(y_test_OOB, predictions_DV_FS_S_tree)
rec_DV_FS_S_tree = recall_score(y_test_OOB,predictions_DV_FS_S_tree)
f1_DV_FS_S_tree = f1_score(y_test_OOB,predictions_DV_FS_S_tree)

model =  pd.DataFrame([['Tree', 'DV_S_FS', accy_DV_FS_S_tree, KAPPA_DV_FS_S_tree, ROC_DV_FS_S_tree, prec_DV_FS_S_tree, rec_DV_FS_S_tree, f1_DV_FS_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[561]:


#KNN
predictions_DV_FS_S_KNN = modelKNN_DV_FS_S.predict(X_test_DV_FS_S)
accy_DV_FS_S_KNN = accuracy_score(y_test_DV_FS_S,predictions_DV_FS_S_KNN)
KAPPA_DV_FS_S_KNN = cohen_kappa_score(y_test_DV_FS_S, predictions_DV_FS_S_KNN)
ROC_DV_FS_S_KNN = roc_auc_score(y_test_OOB,predictions_DV_FS_S_KNN)
prec_DV_FS_S_KNN = precision_score(y_test_OOB, predictions_DV_FS_S_KNN)
rec_DV_FS_S_KNN = recall_score(y_test_OOB,predictions_DV_FS_S_KNN)
f1_DV_FS_S_KNN = f1_score(y_test_OOB,predictions_DV_FS_S_KNN)

model =  pd.DataFrame([['KNN', 'DV_S_FS', accy_DV_FS_S_KNN, KAPPA_DV_FS_S_KNN, ROC_DV_FS_S_KNN, prec_DV_FS_S_KNN, rec_DV_FS_S_KNN, f1_DV_FS_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[562]:


outcome_DV_FS_S = []
model_names_DV_FS_S = []
models_DV_FS_S = [('modelSVC_DV_FS_S', SVC(gamma = 'scale')),
              ('modelRF_DV_FS_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV_FS_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV_FS_S', GradientBoostingClassifier()),
              ('modelTREE_DV_FS_S', tree.DecisionTreeClassifier()),
              ('modelKNN_DV_FS_S', KNeighborsClassifier(n_neighbors=3))]


# In[563]:


for model_name_DV_FS_S, model_DV_FS_S in models_DV_FS_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV_FS_S = model_selection.cross_val_score(model_DV_FS_S, features_DV_FS_S, depVar_DV_FS_S, cv=k_fold_validation, scoring='accuracy')
    outcome_DV_FS_S.append(results_DV_FS_S)
    model_names_DV_FS_S.append(model_name_DV_FS_S)
    output_message_DV_FS_S = "%s| Mean=%f STD=%f" % (model_name_DV_FS_S, results_DV_FS_S.mean(), results_DV_FS_S.std())
    print(output_message_DV_FS_S)


# In[564]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV_FS_S)
ax.set_xticklabels(model_names_DV_FS_S)
plt.show()


# In[ ]:


### Top model is modelGB_DV_FS_S


# In[ ]:


####################################################################################
# Model development -- OOB RFE
####################################################################################


# In[611]:


#Models
modelSVC_OOB_RFE = SVC(gamma = 'scale')
modelRF_OOB_RFE = RandomForestClassifier(n_estimators = 100)
modelLR_OOB_RFE = LinearRegression()
modelLr_OOB_RFE = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB_RFE = GradientBoostingClassifier()
modelTREE_OOB_RFE = tree.DecisionTreeClassifier()
modelKNN_OOB_RFE = KNeighborsClassifier(n_neighbors=3)


# In[612]:


#SVR
modelSVC_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelSVC_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE)) 
modelSVC_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[613]:


#Random Forest
modelRF_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelRF_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE))
modelRF_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[614]:


#Linear Regression
modelLR_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelLR_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE))
modelLR_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[615]:


#Logistic Regression
modelLr_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelLr_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE))
modelLr_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[616]:


#Gradient Boosting
modelGB_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelGB_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE))
modelGB_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[617]:


#Decision Tree
modelTREE_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelTREE_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE)) 
modelTREE_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[618]:


#KNN
modelKNN_OOB_RFE.fit(X_train_OOB_RFE,y_train_OOB_RFE)
print(cross_val_score(modelKNN_OOB_RFE, X_train_OOB_RFE, y_train_OOB_RFE)) 
modelKNN_OOB_RFE.score(X_train_OOB_RFE,y_train_OOB_RFE)


# In[555]:


####################################################################################
# Evaluating the Results -- OOB RFE
####################################################################################


# In[619]:


#SVC
predictions_OOB_RFE_SVC = modelSVC_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_SVC = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_SVC)
KAPPA_OOB_RFE_SVC = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_SVC)
ROC_OOB_RFE_SVC = roc_auc_score(y_test_OOB,predictions_OOB_RFE_SVC)
prec_OOB_RFE_SVC = precision_score(y_test_OOB, predictions_OOB_RFE_SVC)
rec_OOB_RFE_SVC = recall_score(y_test_OOB,predictions_OOB_RFE_SVC)
f1_OOB_RFE_SVC = f1_score(y_test_OOB,predictions_OOB_RFE_SVC)

model =  pd.DataFrame([['SVC', 'OOB_RFE', accy_OOB_RFE_SVC, KAPPA_OOB_RFE_SVC, ROC_OOB_RFE_SVC, prec_OOB_RFE_SVC, rec_OOB_RFE_SVC, f1_OOB_RFE_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[620]:


#Random Forest
predictions_OOB_RFE_RF = modelRF_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_RF = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_RF)
KAPPA_OOB_RFE_RF = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_RF)
ROC_OOB_RFE_RF = roc_auc_score(y_test_OOB,predictions_OOB_RFE_RF)
prec_OOB_RFE_RF = precision_score(y_test_OOB, predictions_OOB_RFE_RF)
rec_OOB_RFE_RF = recall_score(y_test_OOB,predictions_OOB_RFE_RF)
f1_OOB_RFE_RF = f1_score(y_test_OOB,predictions_OOB_RFE_RF)

model =  pd.DataFrame([['RF', 'OOB_RFE', accy_OOB_RFE_RF, KAPPA_OOB_RFE_RF, ROC_OOB_RFE_RF, prec_OOB_RFE_RF, rec_OOB_RFE_RF, f1_OOB_RFE_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[558]:


#Logistic Regression
predictions_OOB_RFE_Lr = modelLr_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_Lr = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_Lr)
KAPPA_OOB_RFE_Lr = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_Lr)
ROC_OOB_RFE_Lr = roc_auc_score(y_test_OOB,predictions_OOB_RFE_Lr)
prec_OOB_RFE_Lr = precision_score(y_test_OOB, predictions_OOB_RFE_Lr)
rec_OOB_RFE_Lr = recall_score(y_test_OOB,predictions_OOB_RFE_Lr)
f1_OOB_RFE_Lr = f1_score(y_test_OOB,predictions_OOB_RFE_Lr)

model =  pd.DataFrame([['Lr', 'OOB_RFE', accy_OOB_RFE_Lr, KAPPA_OOB_RFE_Lr, ROC_OOB_RFE_Lr, prec_OOB_RFE_Lr, rec_OOB_RFE_Lr, f1_OOB_RFE_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[621]:


#Gradient Boosting
predictions_OOB_RFE_GB = modelGB_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_GB = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_GB)
KAPPA_OOB_RFE_GB = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_GB)
ROC_OOB_RFE_GB = roc_auc_score(y_test_OOB,predictions_OOB_RFE_GB)
prec_OOB_RFE_GB = precision_score(y_test_OOB, predictions_OOB_RFE_GB)
rec_OOB_RFE_GB = recall_score(y_test_OOB,predictions_OOB_RFE_GB)
f1_OOB_RFE_GB = f1_score(y_test_OOB,predictions_OOB_RFE_GB)

model =  pd.DataFrame([['GB', 'OOB_RFE', accy_OOB_RFE_GB, KAPPA_OOB_RFE_GB, ROC_OOB_RFE_GB, prec_OOB_RFE_GB, rec_OOB_RFE_GB, f1_OOB_RFE_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[623]:


#Decision Tree
predictions_OOB_RFE_tree = modelTREE_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_tree = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_tree)
KAPPA_OOB_RFE_tree = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_tree)
ROC_OOB_RFE_tree = roc_auc_score(y_test_OOB,predictions_OOB_RFE_tree)
prec_OOB_RFE_tree = precision_score(y_test_OOB, predictions_OOB_RFE_tree)
rec_OOB_RFE_tree = recall_score(y_test_OOB,predictions_OOB_RFE_tree)
f1_OOB_RFE_tree = f1_score(y_test_OOB,predictions_OOB_RFE_tree)

model =  pd.DataFrame([['Tree', 'OOB_RFE', accy_OOB_RFE_tree, KAPPA_OOB_RFE_tree, ROC_OOB_RFE_tree, prec_OOB_RFE_tree, rec_OOB_RFE_tree, f1_OOB_RFE_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[622]:


#KNN
predictions_OOB_RFE_KNN = modelKNN_OOB_RFE.predict(X_test_OOB_RFE)
accy_OOB_RFE_KNN = accuracy_score(y_test_OOB_RFE,predictions_OOB_RFE_KNN)
KAPPA_OOB_RFE_KNN = cohen_kappa_score(y_test_OOB_RFE, predictions_OOB_RFE_KNN)
ROC_OOB_RFE_KNN = roc_auc_score(y_test_OOB,predictions_OOB_RFE_KNN)
prec_OOB_RFE_KNN = precision_score(y_test_OOB, predictions_OOB_RFE_KNN)
rec_OOB_RFE_KNN = recall_score(y_test_OOB,predictions_OOB_RFE_KNN)
f1_OOB_RFE_KNN = f1_score(y_test_OOB,predictions_OOB_RFE_KNN)

model =  pd.DataFrame([['KNN', 'OOB_RFE', accy_OOB_RFE_KNN, KAPPA_OOB_RFE_KNN, ROC_OOB_RFE_KNN, prec_OOB_RFE_KNN, rec_OOB_RFE_KNN, f1_OOB_RFE_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[624]:


outcome_OOB_RFE = []
model_names_OOB_RFE = []
models_OOB_RFE = [('modelSVC_OOB_RFE', SVC(gamma = 'scale')),
              ('modelRF_OOB_RFE', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB_RFE', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB_RFE', GradientBoostingClassifier()),
              ('modelTREE_OOB_RFE', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB_RFE', KNeighborsClassifier(n_neighbors=3))]


# In[625]:


for model_name_OOB_RFE, model_OOB_RFE in models_OOB_RFE:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB_RFE = model_selection.cross_val_score(model_OOB_RFE, features_OOB_RFE, depVar_OOB_RFE, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB_RFE.append(results_OOB_RFE)
    model_names_OOB_RFE.append(model_name_OOB_RFE)
    output_message_OOB_RFE = "%s| Mean=%f STD=%f" % (model_name_OOB_RFE, results_OOB_RFE.mean(), results_OOB_RFE.std())
    print(output_message_OOB_RFE)


# In[626]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB_RFE)
ax.set_xticklabels(model_names_OOB_RFE)
plt.show()


# In[ ]:


# Top Model overall is modelGB_OOB_RFE


# In[ ]:


####################################################################################
# Model development -- OOB RFE Scaled
####################################################################################


# In[627]:


#Models
modelSVC_OOB_RFE_S = SVC(gamma = 'scale')
modelRF_OOB_RFE_S = RandomForestClassifier(n_estimators = 100)
modelLR_OOB_RFE_S = LinearRegression()
modelLr_OOB_RFE_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_OOB_RFE_S = GradientBoostingClassifier()
modelTREE_OOB_RFE_S = tree.DecisionTreeClassifier()
modelKNN_OOB_RFE_S = KNeighborsClassifier(n_neighbors=3)


# In[628]:


#SVR
modelSVC_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelSVC_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S)) 
modelSVC_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[629]:


#Random Forest
modelRF_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelRF_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S))
modelRF_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[630]:


#Linear Regression
modelLR_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelLR_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S))
modelLR_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[631]:


#Logistic Regression
modelLr_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelLr_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S))
modelLr_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[632]:


#Gradient Boosting
modelGB_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelGB_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S))
modelGB_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[633]:


#Decision Tree
modelTREE_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelTREE_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S)) 
modelTREE_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[634]:


#KNN
modelKNN_OOB_RFE_S.fit(X_train_OOB_RFE_S,y_train_OOB_RFE_S)
print(cross_val_score(modelKNN_OOB_RFE_S, X_train_OOB_RFE_S, y_train_OOB_RFE_S)) 
modelKNN_OOB_RFE_S.score(X_train_OOB_RFE_S,y_train_OOB_RFE_S)


# In[635]:


####################################################################################
# Evaluating the Results -- OOB RFE Scaled
####################################################################################


# In[639]:


#SVC
predictions_OOB_RFE_S_SVC = modelSVC_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_SVC = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_SVC)
KAPPA_OOB_RFE_S_SVC = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_SVC)
ROC_OOB_RFE_S_SVC = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_SVC)
prec_OOB_RFE_S_SVC = precision_score(y_test_OOB, predictions_OOB_RFE_S_SVC)
rec_OOB_RFE_S_SVC = recall_score(y_test_OOB,predictions_OOB_RFE_S_SVC)
f1_OOB_RFE_S_SVC = f1_score(y_test_OOB,predictions_OOB_RFE_S_SVC)

model =  pd.DataFrame([['SVC', 'OOB_RFE_S', accy_OOB_RFE_S_SVC, KAPPA_OOB_RFE_S_SVC, ROC_OOB_RFE_S_SVC, prec_OOB_RFE_S_SVC, rec_OOB_RFE_S_SVC, f1_OOB_RFE_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[640]:


#Random Forest
predictions_OOB_RFE_S_RF = modelRF_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_RF = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_RF)
KAPPA_OOB_RFE_S_RF = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_RF)
ROC_OOB_RFE_S_RF = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_RF)
prec_OOB_RFE_S_RF = precision_score(y_test_OOB, predictions_OOB_RFE_S_RF)
rec_OOB_RFE_S_RF = recall_score(y_test_OOB,predictions_OOB_RFE_S_RF)
f1_OOB_RFE_S_RF = f1_score(y_test_OOB,predictions_OOB_RFE_S_RF)

model =  pd.DataFrame([['RF', 'OOB_RFE_S', accy_OOB_RFE_S_RF, KAPPA_OOB_RFE_S_RF, ROC_OOB_RFE_S_RF, prec_OOB_RFE_S_RF, rec_OOB_RFE_S_RF, f1_OOB_RFE_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[641]:


#Logistic Regression
predictions_OOB_RFE_S_Lr = modelLr_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_Lr = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_Lr)
KAPPA_OOB_RFE_S_Lr = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_Lr)
ROC_OOB_RFE_S_Lr = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_Lr)
prec_OOB_RFE_S_Lr = precision_score(y_test_OOB, predictions_OOB_RFE_S_Lr)
rec_OOB_RFE_S_Lr = recall_score(y_test_OOB,predictions_OOB_RFE_S_Lr)
f1_OOB_RFE_S_Lr = f1_score(y_test_OOB,predictions_OOB_RFE_S_Lr)

model =  pd.DataFrame([['Lr', 'OOB_RFE_S', accy_OOB_RFE_S_Lr, KAPPA_OOB_RFE_S_Lr, ROC_OOB_RFE_S_Lr, prec_OOB_RFE_S_Lr, rec_OOB_RFE_S_Lr, f1_OOB_RFE_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[642]:


#Gradient Boosting
predictions_OOB_RFE_S_GB = modelGB_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_GB = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_GB)
KAPPA_OOB_RFE_S_GB = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_GB)
ROC_OOB_RFE_S_GB = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_GB)
prec_OOB_RFE_S_GB = precision_score(y_test_OOB, predictions_OOB_RFE_S_GB)
rec_OOB_RFE_S_GB = recall_score(y_test_OOB,predictions_OOB_RFE_S_GB)
f1_OOB_RFE_S_GB = f1_score(y_test_OOB,predictions_OOB_RFE_S_GB)

model =  pd.DataFrame([['GB', 'OOB_RFE_S', accy_OOB_RFE_S_GB, KAPPA_OOB_RFE_S_GB, ROC_OOB_RFE_S_GB, prec_OOB_RFE_S_GB, rec_OOB_RFE_S_GB, f1_OOB_RFE_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[643]:


#Decision Tree
predictions_OOB_RFE_S_tree = modelTREE_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_tree = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_tree)
KAPPA_OOB_RFE_S_tree = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_tree)
ROC_OOB_RFE_S_tree = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_tree)
prec_OOB_RFE_S_tree = precision_score(y_test_OOB, predictions_OOB_RFE_S_tree)
rec_OOB_RFE_S_tree = recall_score(y_test_OOB,predictions_OOB_RFE_S_tree)
f1_OOB_RFE_S_tree = f1_score(y_test_OOB,predictions_OOB_RFE_S_tree)

model =  pd.DataFrame([['Tree', 'OOB_RFE_S', accy_OOB_RFE_S_tree, KAPPA_OOB_RFE_S_tree, ROC_OOB_RFE_S_tree, prec_OOB_RFE_S_tree, rec_OOB_RFE_S_tree, f1_OOB_RFE_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[644]:


#KNN
predictions_OOB_RFE_S_KNN = modelKNN_OOB_RFE_S.predict(X_test_OOB_RFE_S)
accy_OOB_RFE_S_KNN = accuracy_score(y_test_OOB_RFE_S,predictions_OOB_RFE_S_KNN)
KAPPA_OOB_RFE_S_KNN = cohen_kappa_score(y_test_OOB_RFE_S, predictions_OOB_RFE_S_KNN)
ROC_OOB_RFE_S_KNN = roc_auc_score(y_test_OOB,predictions_OOB_RFE_S_KNN)
prec_OOB_RFE_S_KNN = precision_score(y_test_OOB, predictions_OOB_RFE_S_KNN)
rec_OOB_RFE_S_KNN = recall_score(y_test_OOB,predictions_OOB_RFE_S_KNN)
f1_OOB_RFE_S_KNN = f1_score(y_test_OOB,predictions_OOB_RFE_S_KNN)

model =  pd.DataFrame([['KNN', 'OOB_RFE_S', accy_OOB_RFE_S_KNN, KAPPA_OOB_RFE_S_KNN, ROC_OOB_RFE_S_KNN, prec_OOB_RFE_S_KNN, rec_OOB_RFE_S_KNN, f1_OOB_RFE_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[645]:


outcome_OOB_RFE_S = []
model_names_OOB_RFE_S = []
models_OOB_RFE_S = [('modelSVC_OOB_RFE_S', SVC(gamma = 'scale')),
              ('modelRF_OOB_RFE_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_OOB_RFE_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_OOB_RFE_S', GradientBoostingClassifier()),
              ('modelTREE_OOB_RFE_S', tree.DecisionTreeClassifier()),
              ('modelKNN_OOB_RFE_S', KNeighborsClassifier(n_neighbors=3))]


# In[646]:


for model_name_OOB_RFE_S, model_OOB_RFE_S in models_OOB_RFE_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_OOB_RFE_S = model_selection.cross_val_score(model_OOB_RFE_S, features_OOB_RFE_S, depVar_OOB_RFE_S, cv=k_fold_validation, scoring='accuracy')
    outcome_OOB_RFE_S.append(results_OOB_RFE_S)
    model_names_OOB_RFE_S.append(model_name_OOB_RFE_S)
    output_message_OOB_RFE_S = "%s| Mean=%f STD=%f" % (model_name_OOB_RFE_S, results_OOB_RFE_S.mean(), results_OOB_RFE_S.std())
    print(output_message_OOB_RFE_S)


# In[647]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_OOB_RFE_S)
ax.set_xticklabels(model_names_OOB_RFE_S)
plt.show()


# In[ ]:


# Top model is modelGB_OOB_RFE_S


# In[ ]:


####################################################################################
# Model development -- DV RFE
####################################################################################


# In[648]:


#Models
modelSVC_DV_RFE = SVC(gamma = 'scale')
modelRF_DV_RFE = RandomForestClassifier(n_estimators = 100)
modelLR_DV_RFE = LinearRegression()
modelLr_DV_RFE = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV_RFE = GradientBoostingClassifier()
modelTREE_DV_RFE = tree.DecisionTreeClassifier()
modelKNN_DV_RFE = KNeighborsClassifier(n_neighbors=3)


# In[649]:


#SVR
modelSVC_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelSVC_DV_RFE, X_train_DV_RFE, y_train_DV_RFE)) 
modelSVC_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[650]:


#Random Forest
modelRF_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelRF_DV_RFE, X_train_DV_RFE, y_train_DV_RFE))
modelRF_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[651]:


#Linear Regression
modelLR_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelLR_DV_RFE, X_train_DV_RFE, y_train_DV_RFE))
modelLR_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[652]:


#Logistic Regression
modelLr_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelLr_DV_RFE, X_train_DV_RFE, y_train_DV_RFE))
modelLr_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[653]:


#Gradient Boosting
modelGB_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelGB_DV_RFE, X_train_DV_RFE, y_train_DV_RFE))
modelGB_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[654]:


#Decision Tree
modelTREE_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelTREE_DV_RFE, X_train_DV_RFE, y_train_DV_RFE)) 
modelTREE_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[655]:


#KNN
modelKNN_DV_RFE.fit(X_train_DV_RFE,y_train_DV_RFE)
print(cross_val_score(modelKNN_DV_RFE, X_train_DV_RFE, y_train_DV_RFE)) 
modelKNN_DV_RFE.score(X_train_DV_RFE,y_train_DV_RFE)


# In[555]:


####################################################################################
# Evaluating the Results -- DV RFE
####################################################################################


# In[656]:


#SVC
predictions_DV_RFE_SVC = modelSVC_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_SVC = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_SVC)
KAPPA_DV_RFE_SVC = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_SVC)
ROC_DV_RFE_SVC = roc_auc_score(y_test_DV,predictions_DV_RFE_SVC)
prec_DV_RFE_SVC = precision_score(y_test_DV, predictions_DV_RFE_SVC)
rec_DV_RFE_SVC = recall_score(y_test_DV,predictions_DV_RFE_SVC)
f1_DV_RFE_SVC = f1_score(y_test_DV,predictions_DV_RFE_SVC)

model =  pd.DataFrame([['SVC', 'DV_RFE', accy_DV_RFE_SVC, KAPPA_DV_RFE_SVC, ROC_DV_RFE_SVC, prec_DV_RFE_SVC, rec_DV_RFE_SVC, f1_DV_RFE_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[657]:


#Random Forest
predictions_DV_RFE_RF = modelRF_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_RF = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_RF)
KAPPA_DV_RFE_RF = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_RF)
ROC_DV_RFE_RF = roc_auc_score(y_test_DV,predictions_DV_RFE_RF)
prec_DV_RFE_RF = precision_score(y_test_DV, predictions_DV_RFE_RF)
rec_DV_RFE_RF = recall_score(y_test_DV,predictions_DV_RFE_RF)
f1_DV_RFE_RF = f1_score(y_test_DV,predictions_DV_RFE_RF)

model =  pd.DataFrame([['RF', 'DV_RFE', accy_DV_RFE_RF, KAPPA_DV_RFE_RF, ROC_DV_RFE_RF, prec_DV_RFE_RF, rec_DV_RFE_RF, f1_DV_RFE_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[658]:


#Logistic Regression
predictions_DV_RFE_Lr = modelLr_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_Lr = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_Lr)
KAPPA_DV_RFE_Lr = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_Lr)
ROC_DV_RFE_Lr = roc_auc_score(y_test_DV,predictions_DV_RFE_Lr)
prec_DV_RFE_Lr = precision_score(y_test_DV, predictions_DV_RFE_Lr)
rec_DV_RFE_Lr = recall_score(y_test_DV,predictions_DV_RFE_Lr)
f1_DV_RFE_Lr = f1_score(y_test_DV,predictions_DV_RFE_Lr)

model =  pd.DataFrame([['Lr', 'DV_RFE', accy_DV_RFE_Lr, KAPPA_DV_RFE_Lr, ROC_DV_RFE_Lr, prec_DV_RFE_Lr, rec_DV_RFE_Lr, f1_DV_RFE_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[659]:


#Gradient Boosting
predictions_DV_RFE_GB = modelGB_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_GB = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_GB)
KAPPA_DV_RFE_GB = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_GB)
ROC_DV_RFE_GB = roc_auc_score(y_test_DV,predictions_DV_RFE_GB)
prec_DV_RFE_GB = precision_score(y_test_DV, predictions_DV_RFE_GB)
rec_DV_RFE_GB = recall_score(y_test_DV,predictions_DV_RFE_GB)
f1_DV_RFE_GB = f1_score(y_test_DV,predictions_DV_RFE_GB)

model =  pd.DataFrame([['GB', 'DV_RFE', accy_DV_RFE_GB, KAPPA_DV_RFE_GB, ROC_DV_RFE_GB, prec_DV_RFE_GB, rec_DV_RFE_GB, f1_DV_RFE_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[660]:


#Decision Tree
predictions_DV_RFE_tree = modelTREE_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_tree = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_tree)
KAPPA_DV_RFE_tree = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_tree)
ROC_DV_RFE_tree = roc_auc_score(y_test_DV,predictions_DV_RFE_tree)
prec_DV_RFE_tree = precision_score(y_test_DV, predictions_DV_RFE_tree)
rec_DV_RFE_tree = recall_score(y_test_DV,predictions_DV_RFE_tree)
f1_DV_RFE_tree = f1_score(y_test_DV,predictions_DV_RFE_tree)

model =  pd.DataFrame([['Tree', 'DV_RFE', accy_DV_RFE_tree, KAPPA_DV_RFE_tree, ROC_DV_RFE_tree, prec_DV_RFE_tree, rec_DV_RFE_tree, f1_DV_RFE_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[661]:


#KNN
predictions_DV_RFE_KNN = modelKNN_DV_RFE.predict(X_test_DV_RFE)
accy_DV_RFE_KNN = accuracy_score(y_test_DV_RFE,predictions_DV_RFE_KNN)
KAPPA_DV_RFE_KNN = cohen_kappa_score(y_test_DV_RFE, predictions_DV_RFE_KNN)
ROC_DV_RFE_KNN = roc_auc_score(y_test_DV,predictions_DV_RFE_KNN)
prec_DV_RFE_KNN = precision_score(y_test_DV, predictions_DV_RFE_KNN)
rec_DV_RFE_KNN = recall_score(y_test_DV,predictions_DV_RFE_KNN)
f1_DV_RFE_KNN = f1_score(y_test_DV,predictions_DV_RFE_KNN)

model =  pd.DataFrame([['KNN', 'DV_RFE', accy_DV_RFE_KNN, KAPPA_DV_RFE_KNN, ROC_DV_RFE_KNN, prec_DV_RFE_KNN, rec_DV_RFE_KNN, f1_DV_RFE_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[662]:


outcome_DV_RFE = []
model_names_DV_RFE = []
models_DV_RFE = [('modelSVC_DV_RFE', SVC(gamma = 'scale')),
              ('modelRF_DV_RFE', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV_RFE', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV_RFE', GradientBoostingClassifier()),
              ('modelTREE_DV_RFE', tree.DecisionTreeClassifier()),
              ('modelKNN_DV_RFE', KNeighborsClassifier(n_neighbors=3))]


# In[663]:


for model_name_DV_RFE, model_DV_RFE in models_DV_RFE:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV_RFE = model_selection.cross_val_score(model_DV_RFE, features_DV_RFE, depVar_DV_RFE, cv=k_fold_validation, scoring='accuracy')
    outcome_DV_RFE.append(results_DV_RFE)
    model_names_DV_RFE.append(model_name_DV_RFE)
    output_message_DV_RFE = "%s| Mean=%f STD=%f" % (model_name_DV_RFE, results_DV_RFE.mean(), results_DV_RFE.std())
    print(output_message_DV_RFE)


# In[664]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV_RFE)
ax.set_xticklabels(model_names_DV_RFE)
plt.show()


# In[ ]:


# Top Model overall is modelGB_DV_RFE


# In[ ]:


####################################################################################
# Model development -- DV RFE Scaled
####################################################################################


# In[665]:


#Models
modelSVC_DV_RFE_S = SVC(gamma = 'scale')
modelRF_DV_RFE_S = RandomForestClassifier(n_estimators = 100)
modelLR_DV_RFE_S = LinearRegression()
modelLr_DV_RFE_S = LogisticRegression(solver='lbfgs', max_iter = 4000)
modelGB_DV_RFE_S = GradientBoostingClassifier()
modelTREE_DV_RFE_S = tree.DecisionTreeClassifier()
modelKNN_DV_RFE_S = KNeighborsClassifier(n_neighbors=3)


# In[666]:


#SVR
modelSVC_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelSVC_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S)) 
modelSVC_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[667]:


#Random Forest
modelRF_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelRF_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S))
modelRF_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[668]:


#Linear Regression
modelLR_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelLR_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S))
modelLR_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[669]:


#Logistic Regression
modelLr_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelLr_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S))
modelLr_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[670]:


#Gradient Boosting
modelGB_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelGB_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S))
modelGB_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[671]:


#Decision Tree
modelTREE_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelTREE_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S)) 
modelTREE_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[672]:


#KNN
modelKNN_DV_RFE_S.fit(X_train_DV_RFE_S,y_train_DV_RFE_S)
print(cross_val_score(modelKNN_DV_RFE_S, X_train_DV_RFE_S, y_train_DV_RFE_S)) 
modelKNN_DV_RFE_S.score(X_train_DV_RFE_S,y_train_DV_RFE_S)


# In[673]:


####################################################################################
# Evaluating the Results -- DV RFE Scaled
####################################################################################


# In[674]:


#SVC
predictions_DV_RFE_S_SVC = modelSVC_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_SVC = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_SVC)
KAPPA_DV_RFE_S_SVC = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_SVC)
ROC_DV_RFE_S_SVC = roc_auc_score(y_test_DV,predictions_DV_RFE_S_SVC)
prec_DV_RFE_S_SVC = precision_score(y_test_DV, predictions_DV_RFE_S_SVC)
rec_DV_RFE_S_SVC = recall_score(y_test_DV,predictions_DV_RFE_S_SVC)
f1_DV_RFE_S_SVC = f1_score(y_test_DV,predictions_DV_RFE_S_SVC)

model =  pd.DataFrame([['SVC', 'DV_RFE_S', accy_DV_RFE_S_SVC, KAPPA_DV_RFE_S_SVC, ROC_DV_RFE_S_SVC, prec_DV_RFE_S_SVC, rec_DV_RFE_S_SVC, f1_DV_RFE_S_SVC]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[675]:


#Random Forest
predictions_DV_RFE_S_RF = modelRF_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_RF = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_RF)
KAPPA_DV_RFE_S_RF = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_RF)
ROC_DV_RFE_S_RF = roc_auc_score(y_test_DV,predictions_DV_RFE_S_RF)
prec_DV_RFE_S_RF = precision_score(y_test_DV, predictions_DV_RFE_S_RF)
rec_DV_RFE_S_RF = recall_score(y_test_DV,predictions_DV_RFE_S_RF)
f1_DV_RFE_S_RF = f1_score(y_test_DV,predictions_DV_RFE_S_RF)

model =  pd.DataFrame([['RF', 'DV_RFE_S', accy_DV_RFE_S_RF, KAPPA_DV_RFE_S_RF, ROC_DV_RFE_S_RF, prec_DV_RFE_S_RF, rec_DV_RFE_S_RF, f1_DV_RFE_S_RF]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[676]:


#Logistic Regression
predictions_DV_RFE_S_Lr = modelLr_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_Lr = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_Lr)
KAPPA_DV_RFE_S_Lr = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_Lr)
ROC_DV_RFE_S_Lr = roc_auc_score(y_test_DV,predictions_DV_RFE_S_Lr)
prec_DV_RFE_S_Lr = precision_score(y_test_DV, predictions_DV_RFE_S_Lr)
rec_DV_RFE_S_Lr = recall_score(y_test_DV,predictions_DV_RFE_S_Lr)
f1_DV_RFE_S_Lr = f1_score(y_test_DV,predictions_DV_RFE_S_Lr)

model =  pd.DataFrame([['Lr', 'DV_RFE_S', accy_DV_RFE_S_Lr, KAPPA_DV_RFE_S_Lr, ROC_DV_RFE_S_Lr, prec_DV_RFE_S_Lr, rec_DV_RFE_S_Lr, f1_DV_RFE_S_Lr]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[677]:


#Gradient Boosting
predictions_DV_RFE_S_GB = modelGB_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_GB = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_GB)
KAPPA_DV_RFE_S_GB = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_GB)
ROC_DV_RFE_S_GB = roc_auc_score(y_test_DV,predictions_DV_RFE_S_GB)
prec_DV_RFE_S_GB = precision_score(y_test_DV, predictions_DV_RFE_S_GB)
rec_DV_RFE_S_GB = recall_score(y_test_DV,predictions_DV_RFE_S_GB)
f1_DV_RFE_S_GB = f1_score(y_test_DV,predictions_DV_RFE_S_GB)

model =  pd.DataFrame([['GB', 'DV_RFE_S', accy_DV_RFE_S_GB, KAPPA_DV_RFE_S_GB, ROC_DV_RFE_S_GB, prec_DV_RFE_S_GB, rec_DV_RFE_S_GB, f1_DV_RFE_S_GB]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[678]:


#Decision Tree
predictions_DV_RFE_S_tree = modelTREE_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_tree = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_tree)
KAPPA_DV_RFE_S_tree = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_tree)
ROC_DV_RFE_S_tree = roc_auc_score(y_test_DV,predictions_DV_RFE_S_tree)
prec_DV_RFE_S_tree = precision_score(y_test_DV, predictions_DV_RFE_S_tree)
rec_DV_RFE_S_tree = recall_score(y_test_DV,predictions_DV_RFE_S_tree)
f1_DV_RFE_S_tree = f1_score(y_test_DV,predictions_DV_RFE_S_tree)

model =  pd.DataFrame([['Tree', 'DV_RFE_S', accy_DV_RFE_S_tree, KAPPA_DV_RFE_S_tree, ROC_DV_RFE_S_tree, prec_DV_RFE_S_tree, rec_DV_RFE_S_tree, f1_DV_RFE_S_tree]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[679]:


#KNN
predictions_DV_RFE_S_KNN = modelKNN_DV_RFE_S.predict(X_test_DV_RFE_S)
accy_DV_RFE_S_KNN = accuracy_score(y_test_DV_RFE_S,predictions_DV_RFE_S_KNN)
KAPPA_DV_RFE_S_KNN = cohen_kappa_score(y_test_DV_RFE_S, predictions_DV_RFE_S_KNN)
ROC_DV_RFE_S_KNN = roc_auc_score(y_test_DV,predictions_DV_RFE_S_KNN)
prec_DV_RFE_S_KNN = precision_score(y_test_DV, predictions_DV_RFE_S_KNN)
rec_DV_RFE_S_KNN = recall_score(y_test_DV,predictions_DV_RFE_S_KNN)
f1_DV_RFE_S_KNN = f1_score(y_test_DV,predictions_DV_RFE_S_KNN)

model =  pd.DataFrame([['KNN', 'DV_RFE_S', accy_DV_RFE_S_KNN, KAPPA_DV_RFE_S_KNN, ROC_DV_RFE_S_KNN, prec_DV_RFE_S_KNN, rec_DV_RFE_S_KNN, f1_DV_RFE_S_KNN]],
               columns = ['Model', 'Type', 'Accuracy', 'KAPPA', 'ROC', 'Precision', 'Recall', 'F1 Score'])
model_results = model_results.append(model, ignore_index = True).sort_values(by = ['Accuracy', 'KAPPA', 'ROC'], ascending = False)
model_results


# In[680]:


outcome_DV_RFE_S = []
model_names_DV_RFE_S = []
models_DV_RFE_S = [('modelSVC_DV_RFE_S', SVC(gamma = 'scale')),
              ('modelRF_DV_RFE_S', RandomForestClassifier(n_estimators = 100)),
              ('modelLr_DV_RFE_S', LogisticRegression(solver='lbfgs', max_iter = 4000)),
              ('modelGB_DV_RFE_S', GradientBoostingClassifier()),
              ('modelTREE_DV_RFE_S', tree.DecisionTreeClassifier()),
              ('modelKNN_DV_RFE_S', KNeighborsClassifier(n_neighbors=3))]


# In[681]:


for model_name_DV_RFE_S, model_DV_RFE_S in models_DV_RFE_S:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results_DV_RFE_S = model_selection.cross_val_score(model_DV_RFE_S, features_DV_RFE_S, depVar_DV_RFE_S, cv=k_fold_validation, scoring='accuracy')
    outcome_DV_RFE_S.append(results_DV_RFE_S)
    model_names_DV_RFE_S.append(model_name_DV_RFE_S)
    output_message_DV_RFE_S = "%s| Mean=%f STD=%f" % (model_name_DV_RFE_S, results_DV_RFE_S.mean(), results_DV_RFE_S.std())
    print(output_message_DV_RFE_S)


# In[682]:


fig = plt.figure(figsize=(20,15))
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome_DV_RFE_S)
ax.set_xticklabels(model_names_DV_RFE_S)
plt.show()


# In[ ]:


# Top model is modelGB_DV_RFE_S


# In[ ]:


######################################
# Improve Top Models using Grid Search
######################################


# In[ ]:


#Try to improve top GB model using Grid search and best parameter


# In[51]:


from sklearn.model_selection import GridSearchCV
parameters =[{'loss': ['deviance'], 'learning_rate': [.1, .01, .001, .0001], 'n_estimators': [100, 300, 500, 700, 900, 
                                                                                             1100]},
             {'loss': ['exponential'], 'learning_rate': [.1, .01, .001, .0001], 'n_estimators': [100, 300, 500, 700, 900, 
                                                                                             1100]}]
grid_search = GridSearchCV(estimator = modelGB_OOB_FS_S,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv= 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_OOB_FS_S, y_train_OOB_FS_S)


# In[55]:


best_accuracy = grid_search.best_score_
best_accuracy


# In[62]:


best_parameters = grid_search.best_params_
best_parameters


# In[63]:


#try to improve top M(C) model
parameters =[{'loss': ['deviance'], 'learning_rate': [.1, .008, .01, .02, .03], 'n_estimators': [250, 300, 350, 400]}]
grid_serach = GridSearchCV(estimator = modelGB_OOB_FS_S,
                           param_grid = parameters,                          
                           scoring = 'kappa',
                           cv= 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_OOB_FS_S, y_train_OOB_FS_S)


# In[61]:


best_accuracy = grid_search.best_score_
best_accuracy


# In[ ]:


######################################
# Top Model overall is GB_OOB_FS_S
######################################


# In[683]:


#Variable Importance
modelGB_OOB_FS_S.feature_importances_


# In[684]:


tmp = pd.DataFrame({'Feature': features_OOB_FS_S, 'Feature importance': modelGB_OOB_FS_S.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()


# In[685]:


#Confusion Matrix
confusion_matrix(y_test_OOB_FS_S, predictions_OOB_FS_S_GB)


# In[694]:


######################################
# Try to Improve GradientBoosting Algorithm
######################################

