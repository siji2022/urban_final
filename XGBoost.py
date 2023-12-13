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

x_test=x_test[2056:2056+1,:]
y_test=y_test[2056:2056+1]

x_train=x_train[:,1:] #remove price
x_val=x_val[:,1:]
x_test=x_test[:,1:]

print(f'x_train shape: {x_train.shape}')
MODEL_NAME='XGB_sample'

params = {
    "n_estimators": 500, # 600 doesnt improve
    "max_depth": 10,
    "min_samples_split": 4,
    "learning_rate": 0.05,
    "loss": "squared_error",
}
# # TEST ONLY
all_mae=[]
all_median=[]
all_mape=[]
for idx in range(5):
    # load model using pickle
    import pickle
    with open(f'./uc/final_project_hp/models/{MODEL_NAME}_{idx}.pkl', 'rb') as f:
        model = pickle.load(f)
        y_est_pred=model.predict(x_test)
        mae=mean_absolute_error(y_test,y_est_pred)
        median_ae=metrics.median_absolute_error(y_test,y_est_pred)
        mape=metrics.mean_absolute_percentage_error(y_test,y_est_pred)
        print("The MAE on test set: {:.4f}".format(mae))
        print("The median AE on test set: {:.4f}".format(median_ae))
        print("The MAPE on test set: {:.4f}".format(mape))
        all_mae.append(mae)
        all_median.append(median_ae)
        all_mape.append(mape)
print(f'{MODEL_NAME}:{np.mean(all_mae):.2f}$\pm${np.std(all_mae):.2f}')
print(f'{MODEL_NAME}:{np.mean(all_median):2f}$\pm${np.std(all_median):.2f}')
print(f'{MODEL_NAME}:{np.mean(all_mape):.4f}$\pm${np.std(all_mape):.4f}')
import sys
sys.exit()
# # END OF TEST,STOP THE PROGRAM

all_loss=[]
for idx in range(5):
    for val in [25]:
        # params["max_depth"] = val
        model =  ensemble.GradientBoostingRegressor(**params)
        model.fit(x_train,y_train)
        mae = mean_absolute_error(y_val, model.predict(x_val))
        y_test_pred=model.predict(x_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        all_loss.append(mae)
        # save model using pickle
        import pickle
        with open(f'./uc/final_project_hp/models/{MODEL_NAME}_{idx}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("The MAE on test set: {:.4f}".format(mae))
print(f'{MODEL_NAME}:{np.mean(all_loss)},{np.std(all_loss)}')


loc=x_test[:,10] #pca
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
plt.savefig('./uc/final_project_hp/figures/XGB_scatter_test_result.png',dpi=300)

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(model.staged_predict(x_val)):
    test_score[i] = mean_squared_error(y_val, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    model.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.savefig('./uc/final_project_hp/figures/boosting_train_val_loss.png',dpi=300)
plt.close()

# plot the feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(all_columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    model, x_val, y_val, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(all_columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.savefig('./uc/final_project_hp/figures/boosting_feature_importance.png',dpi=300)



