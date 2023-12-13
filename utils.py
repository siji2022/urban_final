import geopandas as gpd
import matplotlib as plt
import pandas as pd
import os
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from shapely.geometry import Point
import mapclassify as mc
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import metrics

import seaborn as sns
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder
# suppress warnings
import warnings
warnings.filterwarnings('ignore')
import torch 
## print all columns in df
pd.set_option('display.max_columns', None)
#fix sklearn seed
np.random.seed(0)

def data_preprocessing_activelisting(active_listing,sold, normalize=True, standardscale=None,labelencoders=None):
    sold_size=sold.shape[0]
    # fill the sold date with today's date
    active_listing['SOLD DATE']=pd.Timestamp.today()
    # append active listing to sold
    sold=sold.append(active_listing)
    
    return data_preprocessing(sold, normalize=True, standardscale=standardscale,labelencoders=labelencoders,sold_size=sold_size)

def data_preprocessing(sold, normalize=True, standardscale=None,labelencoders=None,sold_size=None):
    # convert column sold date to datetime
    sold['SOLD DATE']=pd.to_datetime(sold['SOLD DATE'])
    
    mortgate_rate=pd.read_csv('./uc/final_project_hp/data/MORTGAGE30US.csv')
    mortgate_rate['DATE']=pd.to_datetime(mortgate_rate['DATE'])
    mortgate_rate.set_index('DATE',inplace=True)
    # mortgate_rate is weekly, backfill to daily
    mortgate_rate=mortgate_rate.resample('D').fillna(method='ffill')
    # create a column mortgate_rate_date, which is the 1 month before sold date
    sold['mortgate_rate_date']=sold['SOLD DATE']-pd.DateOffset(months=1)
    # join the mortgate rate to sold data by date
    sold=sold.merge(mortgate_rate,how='left',left_on='mortgate_rate_date',right_on='DATE')
    # remove the duplicate records
    # sold.drop_duplicates(inplace=True)
    # CAP the mortgate rate to 5%
    # sold.loc[sold['MORTGAGE30US']>6,'MORTGAGE30US']=6
    # over sample the 2022 data
    # sold=sold.append([sold[sold['SOLD DATE'].dt.year==2022]]*2,ignore_index=True)
    # sort the sold data by sold date
    sold=sold.sort_values(by='SOLD DATE')
    sold['YEAR BUILT']=2023-sold['YEAR BUILT']
    # fill missing value with 0
    sold=sold.fillna(0)
    sold.loc[:,'SOLD YEAR Trans']=sold['SOLD DATE'].dt.year
    sold.loc[:,'SOLD DATE Trans']=sold['SOLD DATE'].dt.month/12+sold['SOLD DATE'].dt.year
    # get day of the week
    sold.loc[:,'SOLD DAY OF WEEK']=sold['SOLD DATE'].dt.dayofweek
    # PCA on the lat and lon to get 1d
    pca = PCA(n_components=1)
    pca.fit(sold[['LONGITUDE','LATITUDE']])
    sold['PCA']=pca.transform(sold[['LONGITUDE','LATITUDE']])
    sold['PRICE_TARGET']=sold['PRICE']
    # adjust the price by mortgate rate
    # sold['PRICE']=sold['PRICE']*(1+sold['MORTGAGE30US']/100)

    
    # normalize columns
    normalize_columns=['PRICE','LONGITUDE','LATITUDE','SOLD DATE Trans','BEDS','BATHS','SQUARE FEET','LOT SIZE','$/SQUARE FEET','HOA/MONTH','YEAR BUILT','PCA']
    # use sklearn StandardScaler on normalize_columns
    if standardscale is None:
        standardscale=StandardScaler()
        standardscale.fit(sold[normalize_columns])
    else:
        standardscale=standardscale

    # define the categorical columns
    cat_columns=['PROPERTY TYPE','ZIP OR POSTAL CODE']
    # cat_columns=['PROPERTY TYPE']
    cat_columns_codes=normalize_columns.copy()
    # define x and y
    if sold_size is None:
        # The test data is from 2023 Aug to Nov
        sold_test=sold[sold['SOLD DATE'].dt.year>=2023 ]
        sold_test=sold_test[sold_test['SOLD DATE'].dt.month>=8]
        # sold_train=sold[sold['SOLD DATE'].dt.year<2023]
        # The train data is from 2018 to 2023 July
        sold_train=sold[sold['SOLD DATE']<pd.Timestamp(2023,8,1)]
    else:
        sold_test=sold.iloc[sold_size:,:]
        sold_train=sold.iloc[:sold_size,:]
    sold_test_orig=sold_test.copy()
    sold_train_orig=sold_train.copy()
    if normalize:
        # normalize columns for train dataset
        sold_train.loc[:,normalize_columns]=standardscale.transform(sold_train[normalize_columns])
        # print(sold_train[normalize_columns].describe())
        # sold_train.loc[:,'SOLD DATE']=(sold_train['SOLD DATE'].dt.year-2017)*12+sold_train['SOLD DATE'].dt.month
        # normalize columns for test dataset
        sold_test.loc[:,normalize_columns]=standardscale.transform(sold_test[normalize_columns])
        # sold_test.loc[:,'SOLD DATE']=(sold_test['SOLD DATE'].dt.year-2017)*12+sold_test['SOLD DATE'].dt.month
    sold_train.drop_duplicates(inplace=True)
    x_train=sold_train[normalize_columns].values
    x_test=sold_test[normalize_columns].values

    sold_train.loc[:,cat_columns]=sold_train[cat_columns].astype('object')
    if labelencoders is None:
        labelencoders=[]
        for col in cat_columns:
            le=OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=0.02)
            le.fit(sold_train[[col]])
            labelencoders.append(le)
            # print(f'column {col} has {len(le.categories_[0])} categories') # Property type 7; 
    
    sold_test.loc[:,cat_columns]=sold_test[cat_columns].astype('object')
    for col,le in zip(cat_columns,labelencoders):
        x_train=np.concatenate((x_train,le.transform(sold_train[[col]]).toarray()),axis=1)
        cat_columns_codes.extend(le.get_feature_names_out())
        x_test=np.concatenate((x_test,le.transform(sold_test[[col]]).toarray()),axis=1)
    
    

    y_train=sold_train['PRICE_TARGET'].values
    y_test=sold_test['PRICE_TARGET'].values
    x_test[:,0]=0 # set the price column to be 0 for test dataset
    # concat sold_test_orig and sold_train_orig
    # sold_test_orig=sold_test_orig.append(sold_train_orig)
    return x_train,x_test,y_train,y_test,standardscale,labelencoders,cat_columns_codes,sold_test_orig,sold_train_orig

def moran_scatter_plot(sel,wq,dfs,df_names,lable_ratio=1):
    fig, ax = plt.subplots(1, figsize=(9, 9))
    already_annotated_flag=False
    for df_name, df in zip(df_names,dfs):
        y = df[sel]
        ylag=lps.weights.lag_spatial(wq,y) #apply the spatial weights on the attribute

        ax.plot(y, ylag, '.', label=df_name)
        std=np.std(1-y/ylag) #calculate the ratio of spatial autocorrelation; able the ones away from label_ratio*std
#         print(std)
        if not already_annotated_flag:
            index=0
            for name,y1,ylag1 in zip(df.index,y,ylag):
                ratio=np.abs(1-y1/ylag1)
                if ratio>lable_ratio*std:
                    ax.annotate(name,(y1,ylag1))
                index+=1
            already_annotated_flag=True
                
         # dashed vert at mean of the price
#         plt.vlines(y.mean(), ylag.min(), ylag.max(), linestyle='--')
        #  # dashed horizontal at mean of lagged price
#         plt.hlines(ylag.mean(), y.min(), y.max(), linestyle='--')

        # red line of best fit using global I as slope
        try:
            b, a = np.polyfit(y, ylag, 1)
            plt.plot(y, a + b*y, label=df_name)
        except Exception as e:
            print(e)
    # remove space and special characters in column name
    sel=sel.replace('/','').replace(' ','')
    plt.title('Moran Scatterplot')
    plt.ylabel(f'Spatial Lag of {sel}')
    plt.xlabel(f'{sel}')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig(f'./urban_computing/final_project_hp/figures/moran_scatterplot_{sel}.png',dpi=300)
    plt.close

def sihouette_plot(X,range_n_clusters):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        clusterer.fit(X)
        cluster_labels = clusterer.labels_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        plt.savefig(f'./urban_computing/final_project_hp/figures/silhouette_{n_clusters}.png',dpi=300)

def time_series_trend(sold):
    # analyze the temporal trend in sold data
    # filter the sold data with 'PROPERTY TYPE'=='Condo/Co-op'
    condo=sold[sold['PROPERTY TYPE']=='Condo/Co-op']
    # filter the sold data with 'PROPERTY TYPE'=='Single Family Residential'
    single_family=sold[sold['PROPERTY TYPE']=='Single Family Residential']
    # filter the sold data with 'PROPERTY TYPE'=='Multi-Family (2-4 Unit)'
    multi_family=sold[sold['PROPERTY TYPE']=='Multi-Family (2-4 Unit)']
    # group 'PRICE' by 'SOLD DATE' for conda, single_family and towhouse
    condo_price=condo.groupby('SOLD DATE')['PRICE'].mean().reset_index()
    # convert the 'SOLE DATE' to float
    condo_price['SOLD DATE']=condo_price['SOLD DATE'].dt.year+condo_price['SOLD DATE'].dt.month/12
    single_family_price=single_family.groupby('SOLD DATE')['PRICE'].mean().reset_index()
    single_family_price['SOLD DATE']=single_family_price['SOLD DATE'].dt.year+single_family_price['SOLD DATE'].dt.month/12
    multi_family_price=multi_family.groupby('SOLD DATE')['PRICE'].mean().reset_index()
    multi_family_price['SOLD DATE']=multi_family_price['SOLD DATE'].dt.year+multi_family_price['SOLD DATE'].dt.month/12
    # plot the trend for each type
    sns.regplot(data=single_family_price,x='SOLD DATE',y='PRICE', label='Single-Family Home',marker='.')
    sns.regplot(data=condo_price,x='SOLD DATE',y='PRICE',label='Condo/Co-op',marker='.')
    sns.regplot(data=multi_family_price,x='SOLD DATE',y='PRICE', label='Multi-Family',marker='.')
    plt.legend()
    plt.savefig('./urban_computing/final_project_hp/figures/average_price_sold_date.png')
    plt.close()

def time_series_plot_sales_volumns(df,type=None):
    """plot the time series of sales volumns and median price for the choosen type of property

    Args:
        df (_type_): sold data
        type (_type_, optional): 'Condo/Co-op', 'Single Family Residential'. Defaults to None.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(8,8))
    # add title
    ax1.set_title(f'{type}')
    single_family_price=df.groupby('SOLD DATE')['PRICE'].median().reset_index()
    single_family_price['SOLD DATE']=(single_family_price['SOLD DATE'].dt.year)+single_family_price['SOLD DATE'].dt.month/12+single_family_price['SOLD DATE'].dt.day/360
    ax1.scatter(single_family_price['SOLD DATE'],single_family_price['PRICE'], label='median price',marker='.')
    # plot the moving average
    single_family_price['moving_average']=single_family_price['PRICE'].rolling(window=30).mean()
    ax1.plot(single_family_price['SOLD DATE'],single_family_price['moving_average'], label='moving average',color='red')
    ax1.legend()
    single_family_vol=df.groupby('SOLD DATE')['PRICE'].count().reset_index()
    single_family_vol['SOLD DATE']=(single_family_vol['SOLD DATE'].dt.year)+single_family_vol['SOLD DATE'].dt.month/12+single_family_vol['SOLD DATE'].dt.day/360
    # plot the moving averate
    single_family_vol['moving_average']=single_family_vol['PRICE'].rolling(window=30).mean()
    ax2.scatter(single_family_vol['SOLD DATE'],single_family_vol['PRICE'], label='sales volumn',marker='.')
    ax2.plot(single_family_vol['SOLD DATE'],single_family_vol['moving_average'], label='moving average',color='red')
    ax2.legend()

    plt.savefig(f'./urban_computing/final_project_hp/figures/median_price_sold_date_{type}.png',bbox_inches='tight',dpi=300)

def time_series_plot_by_cca(df, name=None):
    df_price=df.groupby('SOLD DATE')['PRICE'].median().reset_index()
    df_price['SOLD DATE']=(df_price['SOLD DATE'].dt.year)+df_price['SOLD DATE'].dt.month/12+df_price['SOLD DATE'].dt.day/360
    df_price['SOLD DATE']=df_price['SOLD DATE']-df_price['SOLD DATE'].min()
    # define x and y
    y=df_price['PRICE'].values
    x=df_price['SOLD DATE'].values

    coeff, bias = np.polyfit(x.reshape(-1), y.reshape(-1), 1)
    print(f'bias: {bias}; coeff: {coeff}')
    p=sns.regplot(data=df_price,x='SOLD DATE',y='PRICE', marker='.')
    plt.plot(x,bias+coeff*x,color='r',label=f'Linear Regression {name} ')
    plt.title(f'Sold price in {name}')
    plt.legend()
    plt.savefig(f'./urban_computing/final_project_hp/figures/time_price_by_region_{name}.png',dpi=300,bbox_inches='tight')
    plt.close()
    return coeff, bias

def time_series_coeff_by_cca(df):
    df_price=df.groupby('SOLD DATE')['PRICE'].median().reset_index()
    df_price['SOLD DATE']=(df_price['SOLD DATE'].dt.year)+df_price['SOLD DATE'].dt.month/12+df_price['SOLD DATE'].dt.day/360
    df_price['SOLD DATE']=df_price['SOLD DATE']-df_price['SOLD DATE'].min()
    # define x and y
    y=df_price['PRICE'].values
    x=df_price['SOLD DATE'].values.reshape(-1,1)
    try:
        coeff, bias = np.polyfit(x.reshape(-1), y.reshape(-1), 1)
        return coeff, bias
    except:
        return 0,0
    
def plot_coeff_ts_by_cca(boundaries,fn=None):
    """_summary_

    Args:
        boundaries (_type_): geopandas dataframe with columns 'community','coeff','bias','condo_coeff',
    """
        # plot cca_coeff
    # make 4 subplots
    fig, ax = plt.subplots(2, 2,figsize=(10,10))
    ax = np.ravel(ax)
    # fix color bar range for all subplots
    vmin=-1e5
    vmax=1e5
    missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        },
    boundaries.plot(ax=ax[0], column='coeff',cmap='RdBu_r', legend=True,vmin=vmin,vmax=vmax,missing_kwds=missing_kwds,)
    ax[0].set_title('All Coefficient of CCA')

    boundaries.plot(ax=ax[1], column='condo_coeff',cmap='RdBu_r', legend=True,vmin=vmin,vmax=vmax,missing_kwds=missing_kwds)
    ax[1].set_title('Condo Coefficient of CCA')

    boundaries.plot(ax=ax[2], column='sf_coeff',cmap='RdBu_r', legend=True,vmin=vmin,vmax=vmax,missing_kwds=missing_kwds)
    ax[2].set_title('SF Coefficient of CCA')

    boundaries.plot(ax=ax[3], column='mf_coeff',cmap='RdBu_r', legend=True,vmin=vmin,vmax=vmax,missing_kwds=missing_kwds)
    ax[3].set_title('MF Coefficient of CCA')

    plt.savefig(f'./uc/final_project_hp/figures/cca_coeff_{fn}.png',dpi=300,bbox_inches='tight')
    plt.close()

# edge definition
def edge_exist_dist(a,b):
    if torch.abs(a[7]-b[7])<0.1:
        return True
    else:
        return False
    
def edge_exist_temporal(a,b):
    if torch.abs(a[8]-b[8])<1.0 and torch.abs(a[7]-b[7])<0.2:
        return True
    else:
        return False
    
# create torch_geometric dataset
# build edges
def build_edges(x,SAMPLE_RATE=100):
    # x: batch,seq,feature
    # edges: 2,seq
    batch_size=x.size()[0]
    edges_dist=[]
    edges_temporal=[]
    i=0
    # check unique nodes, store used nodes in used_nodes set
    used_nodes=set()
    
    while i < batch_size:
        # add i into used_nodes
        used_nodes.add(i)
        if i%5000==0:
            print(f'finish {i} out of {batch_size}')
        j=i+1
        while j<batch_size:
            if edge_exist_dist(x[i,:],x[j,:]):
                edges_dist.append([i,j])
                edges_dist.append([j,i])
            if edge_exist_temporal(x[i,:],x[j,:]):
                edges_temporal.append([i,j])
                edges_temporal.append([j,i])
            # random increase j size to reduce the computation
            j=j+np.random.randint(1,SAMPLE_RATE)
        
        i=i+np.random.randint(1,SAMPLE_RATE)
        while i in used_nodes:
            i=i+np.random.randint(1,SAMPLE_RATE)

        
    edges_dist=torch.tensor(edges_dist).t().contiguous()
    edges_temporal=torch.tensor(edges_temporal).t().contiguous()
    print(f'edges dist shape: {edges_dist.shape}, total scaned nodes: {len(used_nodes)}')
    print(f'edges temporal shape: {edges_temporal.shape}')
    return edges_dist, edges_temporal