# EDA of sold data
import geopandas as gpd
import matplotlib as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from shapely.geometry import Point
import mapclassify as mc
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error, silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import ensemble, metrics,svm
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from utils import data_preprocessing, sihouette_plot,moran_scatter_plot

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 160
import esda
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import libpysal as lps
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import mapclassify as mc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import pickle
from utils import *

fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
a_fn='./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv'
fn_bm='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
sold=pd.read_csv(fn)
active_listing=pd.read_csv(a_fn)
sold_bm=pd.read_csv(fn_bm)

x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,_=data_preprocessing(sold_bm)
x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,_=data_preprocessing(sold,standardscale=standardscale, labelencoders=labelencoders)

# split x_train and y_train into train and validation
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.05, random_state=0)

x_train=x_train[:,1:] #remove price
x_val=x_val[:,1:]
x_test=x_test[:,1:]

# PCA to reduce dimension
pca = PCA(n_components=4)
pca.fit(x_train)
x_train=pca.transform(x_train)
x_val=pca.transform(x_val)
x_test=pca.transform(x_test)

print(f'x_train shape: {x_train.shape}')
MODEL_NAME='SVR_sample'

params = {
    'kernel':'poly',
    'epsilon': 0.1,
    'C':10000,
    'tol':1e-5,
}
all_loss=[]
for idx in range(1):
    for val in [2]:
        # params["min_samples_split"] = val
        model =  svm.SVR(**params)
        model.fit(x_train,y_train)
        mae = mean_absolute_error(y_val, model.predict(x_val))
        print("The MAE on val set: {:.4f}".format(mae))
        y_test_pred=model.predict(x_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        all_loss.append(mae)
        # save model using pickle
        import pickle
        with open(f'./uc/final_project_hp/models/{MODEL_NAME}_{idx}.pkl', 'wb') as f:
            pickle.dump(model, f)
       
print(f'{MODEL_NAME}:{np.mean(all_loss)},{np.std(all_loss)}')


loc=x_test[:,0] #pca
# make two subplots: top and buttom
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(8,8))
# plot the true value and pred on the first subplot
ax1.plot(loc,y_test,'o',label='true')
ax1.plot(loc,y_test_pred,'o',label='est')
ax1.set_xlabel('location')
ax1.legend()
# plot the error on the second subplot
ax2.plot(loc,y_test_pred-y_test,'o',label='error')
ax2.set_xlabel('location')
ax2.legend()
plt.savefig(F'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result.png',dpi=300)

# test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
# for i, y_pred in enumerate(model.staged_predict(x_val)):
#     test_score[i] = mean_squared_error(y_val, y_pred)







