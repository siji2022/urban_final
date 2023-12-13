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
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

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
from utils import * 

# read the sold data
fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
# fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_sample.csv'
sold=pd.read_csv(fn)
# normalize columns
normalize_columns=['BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH','PCA']
x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns=data_preprocessing(sold, False)

# split x_train and y_train into train and validation
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1, random_state=0)
print(f'x_train shape: {x_train.shape}')
# dt_score=[]
# max_depths_range=range(2,21)
# for depth in max_depths_range:
#     # use deceision tree to fit the data in regression
#     dtmodel = DecisionTreeRegressor(random_state=0, max_depth=depth)
#     dtmodel.fit(x_train,y_train)
#     # use MAE to score the model
#     y_val_pred=dtmodel.predict(x_val)
#     dt_score.append(mean_absolute_error(y_val,y_val_pred))

# # plot the score
# plt.plot(max_depths_range,dt_score,label='DT score')
# plt.xlabel('max depth')
# plt.ylabel('regression score')
# plt.legend()
# plt.savefig('./urban_computing/final_project_hp/figures/dt_score.png',dpi=300)
# plt.close()

# use the best max_depth to fit the model
# best_max_depth=max_depths_range[np.argmin(dt_score)]
best_max_depth=19
print(f'best max depth is {best_max_depth}')
dtmodel = DecisionTreeRegressor(random_state=0, max_depth=best_max_depth)
dtmodel.fit(x_train,y_train)
y_test_pred=dtmodel.predict(x_test)
# use MAE to score the model
print(mean_absolute_error(y_test,y_test_pred))
# print the feature importance
forest_importances = pd.Series(dtmodel.feature_importances_, index=all_columns)
# sort the feature importance
forest_importances.sort_values(ascending=False,inplace=True)
# choose the top 10 features
forest_importances=forest_importances[:10]
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using Decision Tree")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('./urban_computing/final_project_hp/figures/dt_feature_importance.png',dpi=300)


loc=x_test[:,7] # PCA
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
plt.savefig('./urban_computing/final_project_hp/figures/DT_scatter_test_result.png',dpi=300)




