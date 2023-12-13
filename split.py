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
from shapely.geometry import Point,MultiPoint
import mapclassify as mc
from shapely.ops import voronoi_diagram
from shapely.ops import cascaded_union
from shapely.plotting import plot_polygon, plot_points


boundaries_fp='./urban_computing/final_project_hp/data/geo_export.shp'
boundaries=gpd.read_file(boundaries_fp)
# join geometry into one polygon
chicago = gpd.GeoSeries(cascaded_union(boundaries.geometry))
fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
# fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_sample.csv'
sold=pd.read_csv(fn)
# convert column sold date to datetime
sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
# fill missing value with 0
sold=sold.fillna(0)
# convert pd to gpd
sold['geometry'] = sold.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
# print(sold.describe())
# conver pd to geopandas
sold = GeoDataFrame(sold, geometry='geometry')


# fix seed
np.random.seed(0)
km=KMeans(n_clusters=150)
km.fit(sold[['LONGITUDE','LATITUDE']])
sold['cluster']=km.labels_
centers=km.cluster_centers_

ax = plt.gca()
#plots boundaries
# boundaries.plot(ax=ax)
# plots sold data with cluster
sold.plot(ax=ax,markersize=1,c=sold['cluster'])
# plot kmeans boundary
# plt.scatter(centers[:,0],centers[:,1], marker='s', s=100)
mp=MultiPoint(centers)
vor = voronoi_diagram(mp,chicago.iloc[0])
regions=vor.geoms
# for p in regions:
#     plt.plot(*p.exterior.xy, color='black')
for region in regions:
    plot_polygon(region.intersection(chicago.iloc[0]), ax=ax, add_points=False, color='b')
plt.legend()
plt.savefig('./urban_computing/final_project_hp/figures/boundary.png',dpi=300)
plt.close()
