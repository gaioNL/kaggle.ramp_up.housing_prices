
# coding: utf-8

# # Ramp-up Kaggle  - House Prices: EDA, Regression, Pipeline  

# ### Intro
# The this kernel is to get acquainted with Kaggle & experiment with eda, feature enginering & pipelining.
# 
# Many thanks to the guys in the credits for their great kernels!
# 
# ### Credits
# * [House Prices EDA](https://www.kaggle.com/dgawlik/house-prices-eda) by Dominik Gawlik
# 
# * [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset) by Julien Cohen-Solal
# 
# * [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models) by Alexandru Papiu

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
#import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#warnings.filterwarnings('ignore')


# ## Load the data

# In[2]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ## Preprocessing

# In[3]:

# Check for duplicates
idsUnique = len(set(train.Id))
idsTotal = train.shape[0]
idsDupli = idsTotal - idsUnique
print("TRAIN\n# entries: " + str(idsTotal) + "\n# duplicates: " + str(idsDupli))
idsUnique = len(set(test.Id))
idsTotal = test.shape[0]
idsDupli = idsTotal - idsUnique
print("\nTEST\n# entries: " + str(idsTotal) + "\n# duplicates: " + str(idsDupli))

# Drop Id column
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[4]:

# Log transform the target for official scoring
train.SalePrice = np.log1p(train.SalePrice)
y = train.SalePrice


# Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.

# ### Find & fix missing data

# In[5]:

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[20]:

# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)


# In[21]:

# MiscFeature : data description says NA means "no misc feature"
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
#No extra feature = value of the extra feature is O
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)


# In[22]:

# Alley : data description says NA means "no alley access"
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("No")


# In[23]:

# Fence : data description says NA means "no Fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")


# In[25]:

# FireplaceQu : data description says NA means "no FireplaceQu"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)


# In[31]:

#LotFrontage - It means that the property is not visible from the street as it is in a back lo
# see the following interesting discussion on Zillow - 
#https://www.zillow.com/advice-thread/How-to-value-a-home-with-no-street-frontage/325833/
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna("No")


# In[52]:

#GarageCars,GarageArea
sns.countplot(x="GarageCars",data=train)
print("#GarageCars=0=No Garage : " + str(len(train[train['GarageCars']==0])))


# In[13]:

#GarageArea
sns.countplot(x="GarageArea",data=train)
print("#GarageArea=0=No Garage : " + str(len(train[train['GarageArea']==0])))


# In[54]:

#GarageType,GarageCond,GarageFinish,GarageQual,GarageCars,GarageArea
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)


# In[6]:

# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA \means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")


# In[22]:

#MasVnrArea,MasVnrType
print("#MasVnrType= None: " + str(len(train[train['MasVnrType']=='None'])) )
print("#MasVnrArea= 0: " + str(len(train[train['MasVnrArea']==0])) )
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)


# In[25]:

#Eletrical
sns.countplot(x="Electrical",data=train)


# In[31]:

#Electrical - fill with the most common value = SBrkr standard Circuit Breakers & Romex
train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("SBrkr")


# ### Deal with outliers
# We are dealing with house prices. I will look 1st how properties are distributed against sales prices.

# In[ ]:




# ## Target
# ### Summary

# In[ ]:




# In[6]:

train['SalePrice'].describe()


# ### Target Distribution

# In[7]:

sns.distplot(train['SalePrice'])
#Flexibly plot a univariate distribution of observations.
#This function combines the matplotlib hist function (with automatic calculation of a good default bin size) 
#with the seaborn kdeplot() and rugplot() functions. 
#It can also fit scipy.stats distributions and plot the estimated PDF over the data.
#KDE https://en.wikipedia.org/wiki/Kernel_density_estimation


# In[8]:

y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)


# ### Note
# Target does not follow the normal distribution. Lognormal & Johnson seem to be a better fit.

# ## Features
# ### Qualitative/Quantitative

# In[9]:

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']


# In[10]:

print ("#quantitative" , len(quantitative))
print ("#qualitative" , len(qualitative))


# ### Quantitative features distribution

# In[13]:

f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# ### Note 
# Quantitative features do not follow the normal distr. 
# 
# Reduce skewedness by transforming the variables by taking log(feature + 1) 
# 

# In[23]:

from scipy.stats import skew

qual_feat_train= train.dtypes[train.dtypes != "object"].index
qual_feat_test= test.dtypes[test.dtypes != "object"].index

#calculate skewedness
train_skewed_calc = train[qual_feat_train].apply(lambda x: skew(x.dropna()))
test_skewed_calc = test[qual_feat_test].apply(lambda x: skew(x.dropna()))

#get the indexes of the data to transform
train_skewed_idx = train_skewed_calc[train_skewed_calc > 0.75].index
test_skewed_idx = test_skewed_calc[test_skewed_calc > 0.75].index

#normalize features
train[train_skewed_idx] = np.log1p(train[train_skewed_idx])
test[test_skewed_idx] = np.log1p(test[test_skewed_idx])


# In[28]:

train_skewed_idx


# In[27]:

f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# ### Qualitative features
# 
# **1st Visualize the distribution of each feature values**
# 
# Not standard EDA. I want to understand how diverse is the data

# In[19]:

f = pd.melt(train, value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.countplot, "value")


# In[15]:

sns.countplot(x="HouseStyle",data=train)


# ### Note
# We are looking mostly at 1 or 2 stories houses.
# Looking at the features that are meant to score a particular catacteristic of the house, you can see that the dominating values are "TA=typical/avg" & "Gd=good". This does really add too much info & differentiation at 1st sight. Need to observe whether the features are knocked out by Lasso & random forest.

# **Visualize the distribution of the sales price against the categorical features**

# In[20]:

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")


# **Put together feature distribution & sales price distribution against each feature**
# This should help me getting a better qualitative understanding of which features drive the price

# In[29]:

for feature in qualitative:
    fig, ax =plt.subplots(1,2)
    sns.boxplot(train[feature],train['SalePrice'] ,ax=ax[0],palette="Set3")
    sns.countplot(train[feature], ax=ax[1])
    fig.show()


# In[ ]:



