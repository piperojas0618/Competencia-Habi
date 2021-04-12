#!/usr/bin/env python
# coding: utf-8

# # First import all libraries that are going to be used

# In[26]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# # Install kaggle API and download data from competition

# In[2]:


#!pip install kaggle
#!kaggle competitions download -c el-algoritmo-es-correcto


# # Read csv as pandas DataFrame

# In[3]:


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
example = pd.read_csv('example_submission.csv')


# # Divide each column as dummy, float or string

# In[4]:


dummy_var = ['ascensor','balcon','conjuntocerrado','cuartoservicio','deposito','estudio','gimnasio',
            'halldealcobas','parqueaderovisitantes','piscina','porteria','remodelado','saloncomunal',
            'terraza','vigilancia','vista','zonalavanderia']

float_var = ['area','banos','banoservicio','estrato','garajes','habitaciones','piso','valoradministracion',
            'valorventa']

string_var = ['tiempodeconstruido','vista','tipoinmueble','tiponegocio']


# # Apply general filter. Fillna with 0's for dummy and float variables and dropna.

# In[5]:


train_data[dummy_var] = train_data[dummy_var].fillna(0)
train_data['banos'] = train_data['banos'].fillna(1)
train_data['habitaciones'] = train_data['habitaciones'].fillna(1)
train_data['piso'] = train_data['piso'].fillna(1)
train_data[float_var] = train_data[float_var].fillna(0)
test_data[dummy_var] = test_data[dummy_var].fillna(0)
test_data['banos'] = test_data['banos'].fillna(1)
test_data['habitaciones'] = test_data['banos'].fillna(1)
test_data['piso'] = test_data['piso'].fillna(1)
test_data[float_var] = test_data[float_var].fillna(0)
train_data = train_data.dropna()
test_data = test_data.dropna()
train_data = train_data[train_data['tipoinmueble'] == 'Casa']


# # Specific filters for float variables

# In[6]:


q_up = 0.95
q_down = 0.005


# In[7]:


q_banos = train_data["banos"].quantile(q_up)
q_banos1 = train_data['banos'].quantile(q_down)
train_data = train_data[(train_data["banos"] <= q_banos) & (train_data['banos'] > 0)]
boxplot = train_data.boxplot(column=['banos'])


# In[8]:


q_va = train_data['valoradministracion'].quantile(q_up)
train_data = train_data[train_data["valoradministracion"] <= q_va]
boxplot = train_data.boxplot(column=['valoradministracion'])


# In[9]:


q_vv = train_data['valorventa'].quantile(q_up)
q_vv1 = train_data['valorventa'].quantile(q_down)
train_data = train_data[(train_data["valorventa"] <= q_vv) & (train_data['valorventa'] >= q_vv1)]
boxplot = train_data.boxplot(column=['valorventa'])


# In[10]:


q_a = train_data['area'].quantile(q_up)
q_a1 = train_data['area'].quantile(q_down)
train_data = train_data[(train_data["area"] <= q_a) & (train_data['area'] >= q_a1)]
boxplot = train_data.boxplot(column=['area'])


# In[11]:


q_h = train_data['habitaciones'].quantile(q_up)
q_h1 = train_data['habitaciones'].quantile(q_down)
train_data = train_data[(train_data["habitaciones"] <= q_h) & (train_data['habitaciones'] >= q_h1)]
boxplot = train_data.boxplot(column=['habitaciones'])


# In[12]:


q_g = train_data['garajes'].quantile(q_up)
train_data = train_data[train_data["garajes"] <= q_g]
boxplot = train_data.boxplot(column=['garajes'])


# In[13]:


q_p = train_data['piso'].quantile(q_up)
train_data = train_data[train_data["piso"] <= q_p]
boxplot = train_data.boxplot(column=['piso'])


# In[14]:


hist_vv = train_data['valorventa']
sn.set_style("whitegrid")
sn.distplot(hist_vv)
plt.show()

train_data["valorventa_log"] = np.log(train_data['valorventa'])
hist_vvlog = train_data['valorventa_log']
sn.distplot(hist_vvlog)
plt.show()

test_data['valorventa_log'] = np.log(test_data['valorventa'])


# In[15]:


hist_a = train_data['area']
sn.set_style("whitegrid")
sn.distplot(hist_a)
plt.show()

train_data['area_log'] = np.log(train_data['area'])
hist_alog = train_data['area_log']
sn.distplot(hist_alog)
plt.show()

test_data['area_log'] = np.log(test_data['area'])


# # Modify string variables

# In[16]:


silhouette = []  #Empty list
l = ['latitud','longitud','estrato']
X = train_data[l].to_numpy()
#for n_clusters in range(5, 20):
#    kmeans = KMeans(n_clusters = n_clusters, random_state = 99,init='random')  #Implementation of K - MEANS
#    cluster_labels = kmeans.fit_predict(X)    #Predict the right cluster for each sample. 
#    score = silhouette_score(X, cluster_labels)  #Calculate the silhouette score. 
#    silhouette.append(score)  
#    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

#plt.show()
#silhouette_fig, ax = plt.subplots()
#ax.plot(range(5,20), silhouette)
#ax.set_xlabel('Number of clusters')
#ax.set_ylabel('Silhouette score')
#ax.set_xticks(np.arange(5,20, 1.0))
#silhouette_fig.suptitle("Finding the right number of clusters", weight = 'bold', size = 18)

kmeans = KMeans(n_clusters = 10, random_state = 99,init='random')  #Implementation of K - MEANS
cluster_labels = kmeans.fit_predict(X)    #Predict the right cluster for each sample. 
score = silhouette_score(X, cluster_labels)
print(score)
train_data['cluster_labels'] = cluster_labels


# In[17]:


train_data['vista'] = train_data['vista'].replace(0,'Interior')
test_data['vista'] = test_data['vista'].replace(0,'Interior')
train_data['tiponegocio'] = train_data['tiponegocio'].replace('Venta y arriendo','Venta Y Arriendo')
train_data = train_data[train_data['tiponegocio'] != 'Arriendo']
train_data['tiempodeconstruido'] = train_data['tiempodeconstruido'].replace('ntre 0 y 5 años','Entre 0 y 5 años')
train_data['valor/m2'] = train_data['valorventa'] / train_data['area']
train_data['valor_habi'] = train_data['valor/m2']*0.925
train_data


# # Apply Ordinal Encoder to String columns

# In[18]:


enc = OrdinalEncoder()

X_tn = train_data['tiponegocio'].to_numpy().reshape(len(train_data),1)
X_transform_tn = enc.fit_transform(X_tn)
train_data['tiponegocio_int'] = X_transform_tn

X_v = train_data['vista'].to_numpy().reshape(len(train_data),1)
X_transform_v = enc.fit_transform(X_v)
train_data['vista_int'] = X_transform_v

X_ti = train_data['tipoinmueble'].to_numpy().reshape(len(train_data),1)
X_transform_ti = enc.fit_transform(X_ti)
train_data['tipoinmueble_int'] = X_transform_ti

X_tc = train_data['tiempodeconstruido'].to_numpy().reshape(len(train_data),1)
X_transform_tc = enc.fit_transform(X_tc)
train_data['tiempodeconstruido_int'] = X_transform_tc


# In[19]:


X_ttn = test_data['tiponegocio'].to_numpy().reshape(len(test_data),1)
X_transform_ttn = enc.fit_transform(X_ttn)
test_data['tiponegocio_int'] = X_transform_ttn

X_tv = test_data['vista'].to_numpy().reshape(len(test_data),1)
X_transform_tv = enc.fit_transform(X_tv)
test_data['vista_int'] = X_transform_tv

X_tti = test_data['tipoinmueble'].to_numpy().reshape(len(test_data),1)
X_transform_tti = enc.fit_transform(X_tti)
test_data['tipoinmueble_int'] = X_transform_tti

X_ttc = test_data['tiempodeconstruido'].to_numpy().reshape(len(test_data),1)
X_transform_ttc = enc.fit_transform(X_ttc)
test_data['tiempodeconstruido_int'] = X_transform_ttc


# In[20]:


train_data_2 = train_data.select_dtypes(exclude = 'object')
test_data_2 = test_data.select_dtypes(exclude = 'object')
X_test_cluster = test_data[l].to_numpy()
cluster_labels_test = kmeans.predict(X_test_cluster)
test_data_2['cluster_labels'] = cluster_labels_test
train_data_2 = train_data_2.drop(columns = ['area','id','valorventa','latitud','longitud','estrato'])
test_data_2 = test_data_2.drop(columns = ['area','valorventa','latitud','longitud','estrato'])


# In[21]:


columns = train_data_2.columns.tolist()
train_columns = columns[:-6] + columns[-4:]
test_columns = columns[-5]
X = train_data_2[train_columns].to_numpy()
Y = train_data_2[test_columns].to_numpy()
Y = Y.reshape(Y.shape[0],)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
#Linear regression model
reg = LinearRegression().fit(X_train, y_train)
Y_pred = reg.predict(X_test)
mape = mean_absolute_percentage_error(y_test,Y_pred)
print(mape)


# In[24]:


gsc = GridSearchCV(
    estimator=RandomForestRegressor(),   #Estimator used for grid search
    param_grid={        
        'n_estimators': (100,200,500,1000),
        'max_depth': (5,10,15,None),
        'min_samples_split': (2,3,5,10),
    },   #Parameters of that estimator
    cv=5, scoring='neg_mean_absolute_percentage_error', verbose=0,n_jobs=-1)   #Splitting value and metric for evaluation
    
grid_result = gsc.fit(X_train, y_train)
best_params = grid_result.best_params_   #Best parameters after running grid search
    
rfr = RandomForestRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'],
                            min_samples_split=best_params['min_samples_split'],random_state=1,n_jobs=-1)
                            #Best parameters usage for random forest
rfr.fit(X_train,y_train)
y_pred_r=rfr.predict(X_test)
print(mean_absolute_percentage_error(y_pred_r,y_test))
print(best_params)  


# In[30]:


#gsc = GridSearchCV(
#    estimator=GradientBoostingRegressor(),   #Estimator used for grid search
#    param_grid={        
#        "n_estimators":(100, 200, 500, 1000),
#        "max_depth": (5, 10, 15),
#        "learning_rate": (0.01, 0.05, 0.1),
#    },   #Parameters of that estimator
#    cv=5, scoring='neg_mean_absolute_percentage_error', verbose=0,n_jobs=-1)   #Splitting value and metric for evaluation
    
#grid_result = gsc.fit(X_train, y_train)
#best_params = grid_result.best_params_   #Best parameters after running grid search
gbr = GradientBoostingRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'],
                           learning_rate=best_params['learning_rate'],random_state=0)
                            #Best parameters usage for gradient boosting regressor
gbr.fit(X_train,y_train)
y_pred_gr=gbr.predict(X_test)
print(mean_absolute_percentage_error(y_pred_gr,y_test))
print(best_params)  


# In[32]:


#Test model
columns_test = test_data_2.columns.tolist()
X_pred_cols = columns_test[1:]
X_pred = test_data_2[X_pred_cols].to_numpy()
Y_pred_test = reg.predict(X_pred)
test_data_2['valor_mt2_predicted'] = np.round(Y_pred_test.tolist(),3)
submission = pd.DataFrame()
submission['id'] = test_data_2['id']
submission['valormt2_predicted'] = test_data_2['valor_mt2_predicted']
submission.to_csv('submission.csv',index = False, decimal = '.',sep = ',')
submission


# In[33]:


Y_pred_test_rf = rfr.predict(X_pred)
test_data_2['valor_mt2_predicted'] = np.round(Y_pred_test_rf.tolist(),3)
submission = pd.DataFrame()
submission['id'] = test_data_2['id']
submission['valormt2_predicted'] = test_data_2['valor_mt2_predicted']
submission.to_csv('submission_rf.csv',index = False, decimal = '.',sep = ',')
submission


# In[34]:


Y_pred_test_gbr = gbr.predict(X_pred)
test_data_2['valor_mt2_predicted'] = np.round(Y_pred_test_gbr.tolist(),3)
submission = pd.DataFrame()
submission['id'] = test_data_2['id']
submission['valormt2_predicted'] = test_data_2['valor_mt2_predicted']
submission.to_csv('submission_gbr.csv',index = False, decimal = '.',sep = ',')
submission

