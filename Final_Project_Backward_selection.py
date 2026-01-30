#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
tp = pd.read_csv("test_predictions.csv", encoding='unicode_escape')
ci = pd.read_csv("country_indicators.csv", encoding='unicode_escape')
df = tp.merge(ci, left_on='iso3', right_on='iso3', how='inner')  


# Get probability prediction error

# In[3]:


df['error_transformer'] = np.abs(df.y_true_transformer-df.y_pred_proba_transformer)
df['error_ffnn'] = np.abs(df.y_true_ffnn-df.y_pred_proba_ffnn)
df['error_xgboost'] = np.abs(df.y_true_xgboost-df.y_pred_proba_xgboost)


# First, look for and get progress indicator variables

# In[4]:


df


# In[52]:


ci


# In[5]:


df_cols = pd.DataFrame(df.dtypes, columns=('coldtype',)).reset_index().rename(columns={'index': 'colname'})
df_cols['coldtype'] = df_cols['coldtype'].astype('string')
# adjusting a potentially useful variable that might be considered label
df['fsi_rank'] = df['fsi_rank'].astype('string').str.replace(r'\D', '', regex=True).replace('', pd.NA)
# get list of numeric variables
num_vars = df_cols.query("coldtype=='float64'")['colname'].values
# these could be useful; but, I'm ignoring them for now to keep the demonstration simpler...

import itertools

# Indicator variables are often called "one hot" encodings
def one_hot(df, cols):
    """ One-hot encode given `cols` and add as new columns
        to `df`

        Returns tuple of `df` with new columns and list of
        new column names.
    """
    new_cols = list()
    new_col_names = list()
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each)
        new_cols.append(dummies)
        new_col_names.append(dummies.columns.values)

    df = pd.concat([df]+new_cols, axis=1)
    new_col_names = list(itertools.chain.from_iterable(new_col_names))
    return df, new_col_names

# categorical variables we will turn into indicator ("one hot") variables
#cat_vars = ['fsi_category', 'hdr_hdicode', 'hdr_region',
#            'wbi_income_group', 'wbi_lending_category',
#            'wbi_other_(emu_or_hipc)']

cat_vars = ['hdr_region']

# get one hot encodings
df_oh, oh_cols = one_hot(df, cat_vars)
df_oh = df_oh.drop(columns=cat_vars)

df_oh[['error_transformer','error_ffnn','error_xgboost'] + oh_cols]


# In[6]:


df.hdr_region.value_counts()
#We want to use SSA as base


# In[7]:


reduced_oh_cols = \
[ 'hdr_region_AS',
 'hdr_region_EAP',
 'hdr_region_ECA',
 'hdr_region_LAC',
 'hdr_region_SA']


# Correlation between all variables:

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

def corr_heatmap(df):
    # plot correlation heatmap based on code from:
    # https://medium.com/@nikolh92/helpful-visualisations-for-linear-regression-646a5648ad9d
    sns.set(style="white")
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    #mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(20, 16))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    return sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True,
                       linewidths=.5, annot=False, cbar_kws={"shrink": .5})

corr_heatmap(df_oh[['error_transformer','error_ffnn','error_xgboost'] + reduced_oh_cols])
_ = plt.axhline(y=3, c='k'); plt.axvline(x=3, c='k')


# In[9]:


# This will be used to make the (additional) indicator variables of which model a prediction corresponds to
df_oh['model_transformer'] = df_oh['error_transformer'].astype(str)*0+"transformer"
df_oh['model_ffnn'] = df_oh['error_ffnn'].astype(str)*0+"ffnn"
df_oh['model_xgboost'] = df_oh['error_xgboost'].astype(str)*0+"xgboost"

# The amount of error might change depending on if the prediction is positive or negative
# so this indicates if it was a postive or negative prediction
df_oh['prediction_transformer'] = df.y_true_transformer.astype(int)
df_oh['prediction_ffnn'] = df.y_pred_ffnn.astype(int)
df_oh['prediction_xgboost'] = df.y_true_xgboost.astype(int)

# This "stacks" all the predictions together on top of each other so they can all be analyzed together
design_matrix = \
pd.concat([df_oh[['error_transformer', 'model_transformer', 'prediction_transformer']+reduced_oh_cols].rename(columns={'error_transformer':'error','model_transformer':'model','prediction_transformer':'predicts1'}),
           df_oh[['error_ffnn', 'model_ffnn', 'prediction_ffnn']+reduced_oh_cols].rename(columns={'error_ffnn':'error','model_ffnn':'model','prediction_ffnn':'predicts1'}),
           df_oh[['error_xgboost', 'model_xgboost', 'prediction_xgboost']+reduced_oh_cols].rename(columns={'error_xgboost':'error','model_xgboost':'model','prediction_xgboost':'predicts1'})],
          ignore_index=True)

# 0. design_matrix[reduced_oh_cols] effects at "baseline" (`ffnn` predicts 0)

# 1. design_matrix[reduced_oh_cols]*predicts1 offset changes to "baseline" when prediction is 1
design_matrix.predicts1 # is the intercept offset
for x in reduced_oh_cols:
    design_matrix[x+' X predicts1'] = design_matrix[x]*design_matrix['predicts1']

# 2. design_matrix[reduced_oh_cols]*`transformer`/`xgboost` additional offset changes to "baseline"
# when prediction is made by `transformer`/`xgboost` for any prediction (0 or 1)
design_matrix['transformer'] = (design_matrix['model']=="transformer").astype(int) # intercept offset
design_matrix['xgboost'] = (design_matrix['model']=="xgboost").astype(int) # intercept offset
for x in reduced_oh_cols:
    design_matrix[x+' X transformer'] = design_matrix[x]*design_matrix['transformer']
    design_matrix[x+' X xgboost'] = design_matrix[x]*design_matrix['xgboost']

# 3. design_matrix[reduced_oh_cols]*`transformer_predicts1`/`xgboost_predicts1` 
# additional offset changes to "baseline" for non `ffnn` 1 predictions      
design_matrix['transformer X predicts1'] = design_matrix['transformer']*design_matrix['predicts1']
design_matrix['xgboost X predicts1'] = design_matrix['xgboost']*design_matrix['predicts1']
for x in reduced_oh_cols:
    design_matrix[x+' X transformer X predicts1'] = design_matrix[x]*design_matrix['transformer X predicts1']
    design_matrix[x+' X xgboost X predicts1'] = design_matrix[x]*design_matrix['xgboost X predicts1']

# This is to address the "DataFrame is highly fragmented" warning that's being flagged below
design_matrix = design_matrix.copy()
y = design_matrix['error']
del design_matrix['error']
del design_matrix['model']
design_matrix#.columns


# In[10]:


import statsmodels.api as sm

model_0_variables = design_matrix.columns.tolist()
model_0 = sm.OLS(y, sm.add_constant(design_matrix[model_0_variables].astype(int)))
model_0.fit().summary().tables[-1]


# In[11]:


np.asarray(design_matrix[model_0_variables])


# In[12]:


design_matrix[model_0_variables].astype(int)


# In[13]:


# we'll restrict ourselves to only examing categories that happen at least 15 or more times
design_matrix.loc[:,design_matrix.sum()>14].shape


# In[14]:


model_1_variables = design_matrix.loc[:,design_matrix.sum()>14].columns.tolist()
model_1 = sm.OLS(y, sm.add_constant(design_matrix[model_1_variables].astype(int)))
model_1.fit().summary()


# In[15]:


model_2_variables = model_1_variables.copy()

model_2_variables.remove('hdr_region_SA X transformer') 
model_2_variables.remove('hdr_region_SA')                  
model_2_variables.remove('hdr_region_EAP')
model_2_variables.remove('hdr_region_AS')    
model_2_variables.remove('hdr_region_LAC')          
model_2_variables.remove('hdr_region_LAC X predicts1')    
model_2_variables.remove('hdr_region_AS X predicts1')      
model_2_variables.remove('hdr_region_EAP X xgboost')    
model_2_variables.remove('hdr_region_SA X xgboost')
model_2_variables.remove('hdr_region_ECA X xgboost')
model_2_variables.remove('hdr_region_AS X transformer')    
model_2_variables.remove('hdr_region_AS X xgboost')
model_2_variables.remove('hdr_region_EAP X transformer')    
model_2_variables.remove('hdr_region_LAC X transformer')   
model_2_variables.remove('hdr_region_ECA X transformer')  

model_2 = sm.OLS(y, sm.add_constant(design_matrix[model_2_variables].astype(int)))
model_2.fit().summary()


# Remove the variables that are less than 0.01

# In[16]:


model_3_variables = model_2_variables.copy()
#remove some more irrelevent columns        
model_3_variables.remove('hdr_region_ECA')
model_3_variables.remove('transformer X predicts1')

model_3 = sm.OLS(y, sm.add_constant(design_matrix[model_3_variables].astype(int)))
model_3.fit().summary()


# Remove the variables who's p-value is not 0.000

# In[17]:


model_4_variables = model_3_variables.copy()

model_4_variables.remove('transformer')
model_4_variables.remove('xgboost')

model_4 = sm.OLS(y, sm.add_constant(design_matrix[model_4_variables].astype(int)))
model_4.fit().summary()


# In[18]:


_ = plt.hist(y-model_1.fit().predict())


# In[19]:


_ = plt.hist(y-model_2.fit().predict())


# In[20]:


_ = plt.hist(y-model_3.fit().predict())


# In[21]:


_ = plt.hist(y-model_4.fit().predict())


# In[22]:


_ = plt.plot(model_1.fit().predict()+0.05*np.random.uniform(size=len(y)), y-model_1.fit().predict(), '.')


# In[23]:


_ = plt.plot(model_2.fit().predict()+0.05*np.random.uniform(size=len(y)), y-model_2.fit().predict(), '.')


# In[24]:


_ = plt.plot(model_3.fit().predict()+0.05*np.random.uniform(size=len(y)), y-model_3.fit().predict(), '.')


# In[25]:


_ = plt.plot(model_4.fit().predict()+0.05*np.random.uniform(size=len(y)), y-model_4.fit().predict(), '.')


# In[26]:


#each model_#_variable has type error thus .astype(int) is added
#T-test
# np.random.seed(130)
train_size = 800
data_indices = np.random.choice(design_matrix.index, size=design_matrix.shape[0], replace=False)
train_indices = data_indices[train_size:]
test_indices = data_indices[:train_size]

model_1_train_test_fit = sm.OLS(y[train_indices], sm.add_constant(design_matrix.iloc[train_indices][model_1_variables].astype(int))).fit()
model_1_train_RMSE = ((y[train_indices] - model_1_train_test_fit.predict())**2).mean()**.5
model_1_test_RMSE = ((y[test_indices] - 
                      model_1_train_test_fit.predict(sm.add_constant(design_matrix.iloc[test_indices][model_1_variables].astype(int)))
                     )**2).mean()**.5

model_2_train_test_fit = sm.OLS(y[train_indices], sm.add_constant(design_matrix.iloc[train_indices][model_2_variables].astype(int))).fit()
model_2_train_RMSE = ((y[train_indices] - model_2_train_test_fit.predict())**2).mean()**.5
model_2_test_RMSE = ((y[test_indices] - 
                      model_2_train_test_fit.predict(sm.add_constant(design_matrix.iloc[test_indices][model_2_variables].astype(int)))
                     )**2).mean()**.5

model_3_train_test_fit = sm.OLS(y[train_indices], sm.add_constant(design_matrix.iloc[train_indices][model_3_variables].astype(int))).fit()
model_3_train_RMSE = ((y[train_indices] - model_3_train_test_fit.predict())**2).mean()**.5
model_3_test_RMSE = ((y[test_indices] - 
                      model_3_train_test_fit.predict(sm.add_constant(design_matrix.iloc[test_indices][model_3_variables].astype(int)))
                     )**2).mean()**.5

model_4_train_test_fit = sm.OLS(y[train_indices], sm.add_constant(design_matrix.iloc[train_indices][model_4_variables].astype(int))).fit()
model_4_train_RMSE = ((y[train_indices] - model_4_train_test_fit.predict())**2).mean()**.5
model_4_test_RMSE = ((y[test_indices] - 
                      model_4_train_test_fit.predict(sm.add_constant(design_matrix.iloc[test_indices][model_4_variables].astype(int)))
                     )**2).mean()**.5

import plotly.express as px
px.bar(pd.DataFrame({'RMSE': [model_1_train_RMSE, model_2_train_RMSE, model_3_train_RMSE, model_4_train_RMSE] + 
                             [model_1_test_RMSE, model_2_test_RMSE, model_3_test_RMSE, model_4_test_RMSE],
                     'Score': ['Training']*4+['Testing']*4,
                     'Model': [1,2,3,4]+[1,2,3,4]}), 
       y='RMSE', x='Model', color='Score', barmode='group')


# In[27]:


model_2_train_test_fit.summary().tables[1]


# In[28]:


model_3_train_test_fit.summary().tables[1]


# In[77]:


model_4_train_test_fit.summary().tables[1]


# In[30]:


np.random.seed(1)

continuos_predictors = df[[c for c in df.columns.values if c.startswith('sowc')]].dropna(axis='columns').sample(5, axis=1)
continuos_predictors = (continuos_predictors - continuos_predictors.mean()) / continuos_predictors.std()


# In[31]:


_ = corr_heatmap(df_oh[['error_transformer','error_ffnn','error_xgboost']].join(continuos_predictors))


# In[32]:


continuos_predictors_stacked = pd.concat([continuos_predictors]*3, ignore_index=True)

for x in continuos_predictors.columns:
    continuos_predictors_stacked[x+' X predicts1'] = continuos_predictors_stacked[x]*design_matrix['predicts1']

for x in continuos_predictors.columns:
    continuos_predictors_stacked[x+' X transformer'] = continuos_predictors_stacked[x]*design_matrix['transformer']
    continuos_predictors_stacked[x+' X xgboost'] = continuos_predictors_stacked[x]*design_matrix['xgboost']

for x in continuos_predictors.columns:
    continuos_predictors_stacked[x+' X transformer X predicts1'] = continuos_predictors_stacked[x]*design_matrix['transformer X predicts1']
    continuos_predictors_stacked[x+' X xgboost X predicts1'] = continuos_predictors_stacked[x]*design_matrix['xgboost X predicts1']

model_5 = sm.OLS(y, sm.add_constant(design_matrix[model_3_variables].join(continuos_predictors_stacked)))
model_5.fit().summary()


# In[66]:


train_size = 800
data_indices = np.random.choice(design_matrix.index, size=design_matrix.shape[0], replace=False)
train_indices = data_indices[train_size:]
test_indices = data_indices[:train_size]

model_5_train_test_fit = sm.OLS(y[train_indices], sm.add_constant(design_matrix.iloc[train_indices][model_3_variables].astype(int))).fit()
model_5_train_RMSE = ((y[train_indices] - model_5_train_test_fit.predict())**2).mean()**.5
model_5_test_RMSE = ((y[test_indices] - 
                      model_5_train_test_fit.predict(sm.add_constant(design_matrix.iloc[test_indices][model_3_variables].astype(int)))
                     )**2).mean()**.5


# In[33]:


def predict_likelihood(new_data: pd.DataFrame, model_fit, model_variables):
    """
    Predicts likelihood based on trained linear model.

    Parameters:
    - new_data: pd.DataFrame with the same preprocessed structure and columns as model_variables
    - model_fit: Trained OLS model (e.g., model_4_train_test_fit)
    - model_variables: List of variable names used in the model

    Returns:
    - predictions: pd.Series of predicted values (interpreted as likelihood)
    """
    # Ensure columns match and convert to int if needed
    X = new_data[model_variables].astype(int)
    X = sm.add_constant(X)
    predictions = model_fit.predict(X)
    return predictions.clip(0, 1)  # clip values to be between 0 and 1


# In[61]:


random_indices = np.random.choice(df.index, size=19, replace=False)
new_samples = design_matrix.iloc[[56, 40, 68, 37, 74, 62, 9, 5, 86, 61, 18, 43, 2, 11, 3, 7, 83, 48, 19]
]
design_matrix.iloc[random_indices]


# In[69]:


predicted_likelihoods = predict_likelihood(new_samples, model_3_train_test_fit, model_3_variables)
print(predicted_likelihoods)


# In[ ]:




