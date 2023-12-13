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
from sklearn.cluster import DBSCAN, KMeans
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
# drop the rows with missing sold date
sold.dropna(subset=['YEAR BUILT'],inplace=True)
# fill missing value with 0
sold=sold.fillna(0)
sold['YEAR BUILT']=2023-sold['YEAR BUILT']

# convert pd to gpd
sold['geometry'] = sold.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
sold = GeoDataFrame(sold, geometry='geometry')

# Find the boundary of the lake view
# boundaries=boundaries[boundaries['community']=='LAKE VIEW']
# # filter all sold data in lake view
# sold=sold[sold.within(boundaries.iloc[0].geometry)]
sold_orig=sold.copy()

# normalize columns
normalize_columns=['PRICE','BEDS','BATHS','SQUARE FEET','LOT SIZE','YEAR BUILT','$/SQUARE FEET','HOA/MONTH']
sold[normalize_columns]=(sold[normalize_columns]-sold[normalize_columns].mean())/sold[normalize_columns].std()


# plot the sold data
# fig, ax = plt.subplots(figsize=(10,10))
# boundaries.plot(ax=ax,color='white',edgecolor='black')
# sold.plot(ax=ax,markersize=1,color='red')
# plt.savefig('./urban_computing/final_project_hp/figures/lake_view_sold_houses.png',dpi=300)

# adjust ep value so the number of clusters between 3-5
n_clusters_ = 0
ep=1e-3
rate=0.5
range_min=100
range_max=300
while n_clusters_<range_min or n_clusters_>range_max:
    db = DBSCAN(eps=ep, min_samples=3, p=1.).fit(sold[['LONGITUDE','LATITUDE']].values)
    # db = DBSCAN(eps=ep, min_samples=3).fit(sold[['YEAR BUILT']].values)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f'{ep}: {n_clusters_}')
    if n_clusters_< range_min:
        ep=ep*rate+np.random.rand()*1e-4
    elif n_clusters_> range_max:
        ep=ep/rate+np.random.rand()*1e-4
    

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

boundaries.plot(color='white',edgecolor='black')
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
X=sold_orig[['LONGITUDE','LATITUDE']].values
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k
    cluster_center=sold_orig[class_member_mask]['YEAR BUILT'].mean()
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        # markerfacecolor=tuple(col),
        # markeredgecolor="k",
        markersize=1,
        label=f"cluster {cluster_center:.2f}",
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        # markerfacecolor=tuple(col),
        # markeredgecolor="k",
        markersize=1,
    )
plt.legend()
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.savefig('./urban_computing/final_project_hp/figures/dbscan.png',dpi=300)
