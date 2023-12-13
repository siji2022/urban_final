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

import seaborn as sns

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

boundaries_fp='./urban_computing/final_project_hp/data/geo_export.shp'
boundaries=gpd.read_file(boundaries_fp)

# read the sold data
fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
sold=pd.read_csv(fn)
# convert column sold date to datetime
sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
# fill missing value with 0
sold=sold.fillna(0)
sold['YEAR BUILT']=2023-sold['YEAR BUILT']
# normalize columns
normalize_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH']
sold[normalize_columns]=(sold[normalize_columns]-sold[normalize_columns].mean())/sold[normalize_columns].std()
# convert pd to gpd
sold['geometry'] = sold.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
# print(sold.describe())
# conver pd to geopandas
sold = GeoDataFrame(sold, geometry='geometry')
# Find the boundary of the lake view
boundaries=boundaries[boundaries['community']=='LAKE VIEW']
# filter all sold data in lake view
sold=sold[sold.within(boundaries.iloc[0].geometry)]
sold=sold[sold['PROPERTY TYPE']=='Single Family Residential']
sold_orig=sold.copy()

# normalize columns
normalize_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH','LONGITUDE','LATITUDE']
sold[normalize_columns]=(sold[normalize_columns]-sold[normalize_columns].mean())/sold[normalize_columns].std()


# corr_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH','ZIP OR POSTAL CODE']
# spatial weights
# wq =  lps.weights.Queen.from_dataframe(sold)
# wq.transform = 'r'
# for col in corr_columns:
#     moran_scatter_plot(col,wq,[sold],['5yrs'],lable_ratio=2)



km=KMeans(n_clusters=10)
km.fit(sold[['LONGITUDE','LATITUDE','PRICE']])
sold['cluster']=km.labels_

ax = plt.gca()
#plots boundaries
boundaries.plot(ax=ax)
# plots sold data with cluster
sold_orig.plot(ax=ax,markersize=1,c=sold['cluster'])
plt.legend()
plt.savefig('./urban_computing/final_project_hp/figures/km_lakeview.png',dpi=300)
plt.close()
print(km.inertia_)

# range_n_clusters = [77,150]
# X=sold[['LONGITUDE','LATITUDE']].values
# sihouette_plot(X=X,range_n_clusters=range_n_clusters)
