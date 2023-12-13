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

from utils import *

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

boundaries_fp='./uc/final_project_hp/data/geo_export.shp'
boundaries=gpd.read_file(boundaries_fp)

# read the sold data
fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
# fn='./urban_computing/final_project_hp/data/sold_chicago_5yrs_sample.csv'
sold=pd.read_csv(fn)
# convert column sold date to datetime
sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
# fill missing value with 0
sold=sold.fillna(0)
sold['geometry'] = sold.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
sold = GeoDataFrame(sold, geometry='geometry')

# if 'PROPERTY TYPE' == 'Co-op', change to 'Condo/Co-op'
sold.loc[sold['PROPERTY TYPE']=='Co-op','PROPERTY TYPE']='Condo/Co-op'
sold.loc[sold['PROPERTY TYPE']=='Ranch','PROPERTY TYPE']='Other'
sold.loc[sold['PROPERTY TYPE']=='Unknown','PROPERTY TYPE']='Other'
sold.loc[sold['PROPERTY TYPE']=='Mobile/Manufactured Home','PROPERTY TYPE']='Other'


print(sold['PROPERTY TYPE'].value_counts())

# filter sold data by SOLD DATE after 2020
sold=sold[sold['SOLD DATE'].dt.year>=2021]

# merge the sold data with boundaries by the lat and lon
sold=gpd.sjoin(sold,boundaries,how='left',op='within')

#join sold data to boundaries, aggregate the price by community
boundaries=boundaries.merge(sold.groupby('community')['PRICE'].mean().reset_index(),on='community',how='left')
# plot boundary map, show the average price of each community, cmap use color map
boundaries.plot(column='PRICE',legend=True,figsize=(10,10),cmap='coolwarm')


# filter the sold data with 'PROPERTY TYPE'=='Condo/Co-op'
condo=sold[sold['PROPERTY TYPE']=='Condo/Co-op']
# filter the sold data with 'PROPERTY TYPE'=='Single Family Residential'
single_family=sold[sold['PROPERTY TYPE']=='Single Family Residential']
# filter the sold data with 'PROPERTY TYPE'=='Multi-Family (2-4 Unit)'
multi_family=sold[sold['PROPERTY TYPE']=='Multi-Family (2-4 Unit)']


for community in boundaries['community'].unique():
    # all properties combined
    sold_community=sold[sold['community']==community]
    # for individual property type
    condo_community=condo[condo['community']==community]
    single_family_community=single_family[single_family['community']==community]
    multi_family_community=multi_family[multi_family['community']==community]
    
    coeff, bias = time_series_coeff_by_cca(sold_community)
    boundaries.loc[boundaries['community']==community,'coeff']=coeff
    boundaries.loc[boundaries['community']==community,'bias']=bias

    coeff, bias = time_series_coeff_by_cca(condo_community)
    boundaries.loc[boundaries['community']==community,'condo_coeff']=coeff
    boundaries.loc[boundaries['community']==community,'condo_bias']=bias

    coeff, bias = time_series_coeff_by_cca(single_family_community)
    boundaries.loc[boundaries['community']==community,'sf_coeff']=coeff
    boundaries.loc[boundaries['community']==community,'sf_bias']=bias

    coeff, bias = time_series_coeff_by_cca(multi_family_community)
    boundaries.loc[boundaries['community']==community,'mf_coeff']=coeff
    boundaries.loc[boundaries['community']==community,'mf_bias']=bias

# read boundaries from csv file as geopandas
# boundaries_with_coeff=pd.read_csv('./urban_computing/final_project_hp/data/boundaries_coeff.csv')
# boundaries=boundaries.merge(boundaries_with_coeff[['community','coeff','bias','condo_coeff',
#                                                    'condo_bias','sf_coeff','sf_bias','mf_coeff','mf_bias']],on='community',how='left')


plot_coeff_ts_by_cca(boundaries,'afterorequal2021')
# save boundaries's coeff, drop the geometry column
boundaries=boundaries.drop(columns=['geometry'])
boundaries.to_csv('./uc/final_project_hp/data/boundaries_coeff_afterequal2021.csv',index=False)


# backup
# time_series_plot_sales_volumns(single_family,'Single Family ')
# time_series_plot_sales_volumns(condo,'Condo')
# time_series_plot_sales_volumns(multi_family,'Multi-Family ')
# time_series_plot_sales_volumns(sold,'All')
