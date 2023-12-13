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
from sklearn import ensemble, metrics
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from utils import sihouette_plot,moran_scatter_plot

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
from sklearn.neighbors import KNeighborsRegressor
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

print(f'x_train shape: {x_train.shape}')
MODEL_NAME='KNN'
# KNN
# knn_val_score=[]
# for n_neighbors in range(3,15):
#     model=KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
#     model.fit(x_train,y_train)
#     y_val_pred=model.predict(x_val)
#     knn_val_score.append(mean_absolute_error(y_val,y_val_pred))

# #plot the score
# plt.plot(range(3,15),knn_val_score,label='KNN score')
# plt.xlabel('n_neighbors')
# plt.ylabel('regression score')
# plt.title('KNN score vs n_neighbors')
# plt.savefig('./urban_computing/final_project_hp/figures/KNN_score.png',dpi=300)
# PCA to reduce dimension
pca = PCA(n_components=4)
pca.fit(x_train)
x_train=pca.transform(x_train)
x_val=pca.transform(x_val)
x_test=pca.transform(x_test)

results=[]
for i in range(1):
    model=KNeighborsRegressor(n_neighbors=5, weights='distance',leaf_size=40)
    model.fit(x_train,y_train)

    y_val_pred=model.predict(x_val)
    mae=mean_absolute_error(y_val,y_val_pred)
    # save model using pickle
    import pickle
    with open(f'./uc/final_project_hp/models/{MODEL_NAME}_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("The MAE on val set: {:.4f}".format(mae))

    y_est_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test,y_est_pred)
    results.append(mae)
    median_ae=metrics.median_absolute_error(y_test,y_est_pred)
    mape=metrics.mean_absolute_percentage_error(y_test,y_est_pred)
    print("The MAE on test set: {:.4f}".format(mae))
    print("The median AE on test set: {:.4f}".format(median_ae))
    print("The MAPE on test set: {:.4f}".format(mape))
print(f'KNN MAE: {np.mean(results)}, std: {np.std(results)}')

# loc=x_test[:,10]
# # make two subplots: top and buttom
# fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(8,8))
# # plot the true value and pred on the first subplot
# ax1.plot(loc,y_test,'o',label='true')
# ax1.plot(loc,y_est_pred,'o',label='est')
# ax1.set_xlabel('location')
# ax1.legend()
# # plot the error on the second subplot
# ax2.plot(loc,y_est_pred-y_test,'o',label='error')
# ax2.set_xlabel('location')
# ax2.legend()
# plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result.png',dpi=300)