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
from utils import *
import copy
from torch import Tensor
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 160
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv, SAGEConv,GATConv,GraphConv,GatedGraphConv,TransformerConv,EdgeConv,DynamicEdgeConv
from torch_geometric.nn import global_mean_pool
import torch_geometric as tg
from MyConv import MyConv, MyModel


# fix random seed
torch.manual_seed(0)
np.random.seed(0)

# read the sold data
fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
a_fn='./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv'
fn_bm='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
sold=pd.read_csv(fn)
active_listing=pd.read_csv(a_fn)
sold_bm=pd.read_csv(fn_bm)

x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,sold_train_orig=data_preprocessing(sold_bm)
x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,sold_train_orig=data_preprocessing(sold,standardscale=standardscale, labelencoders=labelencoders)
# x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=\
# data_preprocessing_activelisting(active_listing,sold,standardscale=standardscale, labelencoders=labelencoders)
# split x_train and y_train into train and validation
# x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.05, random_state=0)
# select the one need to investigate

x_test=x_test[2056:2056+1,:]
y_test=y_test[2056:2056+1]
test_orig_df=pd.concat([test_orig_df.iloc[2056:2056+1],sold_train_orig],axis=0)

print(f'x_test shape: {x_test.shape}')

# transform to torch dataset
X_train = torch.from_numpy(x_train).float()
Y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_test = torch.from_numpy(x_test).float()
Y_test = torch.from_numpy(y_test).float().unsqueeze(1)


train_data = Data( y=Y_train,x=X_train)
test_data = Data(x=X_test,  y=Y_test)

# since the edge is created when model is in train, the dataset can be a normal tensor
train_tensor = torch.utils.data.TensorDataset(X_train, Y_train) 
train_data_loader = torch.utils.data.DataLoader(train_tensor, batch_size=4096*13*2,shuffle=False)



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# device = torch.device("cpu")
MODEL_NAME='GNN'




def test(model,train_data, test_data,crit):
    model.eval()
    loss_all=0
    with torch.no_grad():
        pred=[]
        gt=[]
  
        x=test_data.x.to(device)
        test_size=x.shape[0]
        x_train=train_data.x.to(device)
        y_train=train_data.y.to(device)
        x=torch.cat((x,x_train),dim=0)
        y=test_data.y.to(device)

        temp,edge=model(x,test_size)
        print(edge.shape)
        temp=temp[:test_size]
        y_pred=temp.detach().cpu().numpy()
        y_gt=y.cpu().numpy()
        if len(pred)==0:
            pred=y_pred
            gt=y_gt
        else:
            pred=np.concatenate((pred,y_pred),axis=0)
            gt=np.concatenate((gt,y_gt),axis=0)
        loss = crit(temp, y)
        loss_all=loss_all+loss.item()
        loss_all=loss_all/test_size
    return loss_all,pred,gt,edge




    


        
# Test only
all_loss=[]
for bmidx in ['ab5']:
    model=MyModel(x_train.shape[1], 2048,K=2, device=device).to(device)
    loss=0
    # bmidx=2
    try:
        model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
    except:
        print('Test: no saved model')
    # initialize the loss function using mean squared error
    crit=nn.MSELoss(reduction='sum')
    test_loss,pred,gt,edge=test(model,train_data, test_data, nn.L1Loss(reduction='sum'))
    # test_orig_df['pred']=pred.reshape(-1)
    # test_orig_df.to_csv(f'./uc/final_project_hp/{MODEL_NAME}_test_result.csv')
    
    print(f'test loss: {test_loss:.8E}')
    all_loss.append(test_loss)
print(f'{MODEL_NAME}:{np.mean(all_loss)},{np.std(all_loss)}')
# find the idx of the top 10 largest error
idx=np.argsort(np.abs(pred.reshape(-1)-gt.reshape(-1)))[-20:]
print(f'LARGEST ABS test loss: {test_loss:.8E},\n{idx}, \n{pred[idx].reshape(-1)}, \n{gt[idx].reshape(-1)}')

error_df=test_orig_df.iloc[idx]
error_df['pred']=pred[idx].reshape(-1)



# idx=np.argsort((pred.reshape(-1)-gt.reshape(-1)))[-5:]
# # print(f'Expensive test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
# error_df1=test_orig_df.iloc[idx]
# error_df1['pred']=pred[idx].reshape(-1)

# idx=np.argsort((-pred.reshape(-1)+gt.reshape(-1)))[-5:]
# # print(f'Cheaper test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
# error_df2=test_orig_df.iloc[idx]
# error_df2['pred']=pred[idx].reshape(-1)
# # for id in idx:
# #     print(test_orig_df.iloc[id])
# # concatenate  error_df and error_df1 and save to csv
# error_df=pd.concat([error_df,error_df1,error_df2])
error_df.to_csv(f'./uc/final_project_hp/{MODEL_NAME}_error_df.csv')
compare_df=pd.concat([test_orig_df.iloc[edge[0].cpu().numpy()], test_orig_df.iloc[edge[1].cpu().numpy()]])
compare_df.to_csv(f'./uc/final_project_hp/{MODEL_NAME}_compare_df.csv')

# plot
loc=x_test[:,11] #pca
plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
plt.plot(loc,pred,'o', label='predicted', color='red')
plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result.png')

# plot heatmap
x_loc=x_test[:,1] #long
y_loc=x_test[:,2] #lat
diff=pred-gt

df=pd.DataFrame({'x':x_loc,'y':y_loc,'diff':diff.reshape(-1)})
plt.figure(figsize=(15,5))
# sort the df by diff, largest diff first
# df=df.sort_values(by=['diff'],ascending=False)[:1000]
sns.scatterplot(data=df,x='x',y='y',hue='diff',palette='coolwarm',size='diff')
# plt.plot(loc,pred,'o', label='predicted', color='red')
# plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result_heatmap.png')

# clean GPU memory
torch.cuda.empty_cache()