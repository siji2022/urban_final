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
# from MyConv import MyConv, MyModel
# fix random seed
# torch.manual_seed(0)
# np.random.seed(0)

# read the sold data
fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
a_fn='./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv'
fn_bm='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
sold=pd.read_csv(fn)
active_listing=pd.read_csv(a_fn)
sold_bm=pd.read_csv(fn_bm)

x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,_=data_preprocessing(sold_bm)
x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,_=data_preprocessing(sold,standardscale=standardscale, labelencoders=labelencoders)
# x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=\
# data_preprocessing_activelisting(active_listing,sold,standardscale=standardscale, labelencoders=labelencoders)
# split x_train and y_train into train and validation
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1, random_state=0,shuffle=False)
print(f'x_train shape: {x_train.shape}')

# transform to torch dataset
X_train = torch.from_numpy(x_train).float()
Y_train = torch.from_numpy(y_train).float().unsqueeze(1)
# x_val[:,0]=0 # set the price to 0 for validation data
X_val = torch.from_numpy(x_val).float()
Y_val = torch.from_numpy(y_val).float().unsqueeze(1)
X_test = torch.from_numpy(x_test).float()
Y_test = torch.from_numpy(y_test).float().unsqueeze(1)


train_data = Data( y=Y_train,x=X_train)
val_data = Data(x=X_val, y=Y_val)
test_data = Data(x=X_test,  y=Y_test)

# since the edge is created when model is in train, the dataset can be a normal tensor
train_tensor = torch.utils.data.TensorDataset(X_train, Y_train) 
train_data_loader = torch.utils.data.DataLoader(train_tensor, batch_size=4096*2*2*2,shuffle=False)



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# device = torch.device("cpu")
MODEL_NAME='SAGEConv'


class MyModel(torch.nn.Module):

    def __init__(self, input_size, hs,  K=5, device='cpu') -> None:
        super().__init__()
        self.device=device
        self.K=K
        self.r1_edge_index=None        
        self.r1_edge_attr=None
        self.neg_edge_index=None    
        

        self.fc1=nn.Linear(input_size-1,hs)
     

        self.graph1 = SAGEConv( hs+1, hs)
        self.fc2=   nn.Linear(hs,hs)

        # self.fc2_1=nn.Linear(hs+t,hs)
        # self.fc2_2=nn.Linear(hs,hs)
        # self.fc_after_conv1=nn.Linear(hs>>1,hs>>2)
        self.fc3 = nn.Linear(hs, 1)

        self.dropout = nn.Dropout(0)
        self.counter=0
        # self.fc_class_weight=nn.Linear(8,1) # map the property type into a loss weight 


    def get_params_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def create_edges(self,x,create_edge_node_size=0, start_idx=0, k=5):
        edge_index_start=[]
        edge_index_end=[]
        edge_attrs=[]

        i=0
        # start_idx=0
        # norm x
        x_norm=x/torch.norm(x,dim=1).reshape(-1,1)
        # pdist = nn.PairwiseDistance(p=1)
        while i <create_edge_node_size:
            # get the dot product of all other nodes
            if start_idx==0:
                history_size=max(1000,i)
                dot_product=torch.matmul(x_norm[i,:],x_norm[:history_size].T)
            else:
                dot_product=torch.matmul(x_norm[i,:],x_norm[:].T)
            # dot_product=torch.matmul(x_norm[i,:],x_norm[:].T)
            # dot_product1=F.cosine_similarity(x[i,:].reshape(1,-1),x[:],dim=1)
            # dot_product=pdist(x_norm[i,:].reshape(1,-1),x_norm[:])
            # pick the top 10 neighbors index to add in edge_index
            weights,topK=torch.topk(dot_product[start_idx:],k,largest=True)
            # pick the weights larger than 0.98

            if self.training:
                weights=weights[weights>0.98]
            else:
                weights=weights[weights>0.98]

            topK=topK[:weights.shape[0]]

            # add the edge
            # edge_index_start.append((torch.ones(k)*i).to(device))
            # edge_index_end.append(topK+start_idx)
            edge_index_end.append((torch.ones(weights.shape[0])*i).to(self.device))
            edge_index_start.append(topK+start_idx)
            # softmax weights
            edge_attrs.append(F.softmax(weights,dim=0))
    
            # i=i+np.random.randint(1,self.sample_rate)
            i=i+1
        edge_index=torch.cat((torch.cat(edge_index_start).reshape(-1,1),torch.cat(edge_index_end).reshape(-1,1)),dim=1).long()
        edge_index=edge_index.t().contiguous()
        

        edge_attrs=torch.cat(edge_attrs).reshape(-1,1)
        edge_index,edge_attrs=tg.utils.remove_self_loops(edge_index,edge_attrs)
        # if not self.training:
        # print(f'edge_index: {edge_index.shape}, size: {edge_index.shape[1]/create_edge_node_size}')

        return edge_index,edge_attrs,None


    def forward(self, x,  test_size=0):
    
        if self.training:
            self.counter+=1
        x_orig=x
        
        x_with_price=x_orig
        x_wo_price=x_orig[:,1:]

        # x=torch.tanh(self.fc1(x_wo_price)) # because at test time we don't have the price, so we need to remove the price from the input
        x=self.fc1(x_wo_price) # because at test time we don't have the price, so we need to remove the price from the input
        # check if model in train mode: test mode will always create edge_index; train mode will create edge_index with probability 0.2
        if not self.training or self.r1_edge_index is None :
                
            k1=self.K
            
            if self.training:
                # print(f'create edges {self.counter}')
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x_wo_price,create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x,create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(torch.cat((x,x_with_price[:,-8:]),dim=1),create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                self.r1_edge_index=r1_edge_index
                self.neg_edge_index=neg_edge_index
                self.r1_edge_attr=r1_edge_attr
              
            else: # test only; always create edges
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x_wo_price,create_edge_node_size=test_size,start_idx=test_size,k=k1)
                r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x,create_edge_node_size=test_size,start_idx=test_size,k=k1)
        else:
            r1_edge_index=self.r1_edge_index
            r1_edge_attr=self.r1_edge_attr
            neg_edge_index=self.neg_edge_index
        
        
       
        # r1_edge, r1_attr=tg.utils.remove_self_loops(r1_edge_index,r1_edge_attr)
        r1_edge, r1_attr=r1_edge_index, r1_edge_attr
        weighted_price1=x_with_price[r1_edge[0],0:1] # the price is used to estimate the neighbor price; the same est used at test time
        
        target, src = r1_edge
        x_price1=tg.utils.scatter(weighted_price1, src, dim=0,dim_size=x.shape[0], reduce='mean')

       
        # if self.counter%100==0 :
        #     print(f'\t before x_price : {torch.mean(x_price[:test_size])},{torch.mean(x_price[test_size:])}, {x_price.max()}')
        #     print(f'\t after  x_price1 : {torch.mean(x_price1[:test_size])},{torch.mean(x_price1[test_size:])}')
        #     print(f'\t after  x_price2 : {torch.mean(x_price2[:test_size])},{torch.mean(x_price2[test_size:])}')
        # x_price1=F.relu(self.graph1(x_price1.reshape(-1,1),r1_edge_index,r1_edge_attr))

        x=self.dropout(x)
        x=torch.tanh(x)
        x_i=torch.cat((x,x_price1.reshape(-1,1)),dim=1)        # x is concat with estimated price est
        x_j=torch.cat((x,x_with_price[:,0].reshape(-1,1)),dim=1)        # x is concat with original price est
        # x_i=torch.tanh(x_i)
        # x_j=torch.tanh(x_j)

        x =self.graph1(x_i,r1_edge)
        # batch normalization on the conv output
        # x_conv = F.batch_norm(x_conv, running_mean=None, running_var=None, training=True, momentum=0.1)
        x=F.relu(self.fc2(x))
        # x=F.relu(x)+x_conv
        # x=torch.cat((x,x_conv),dim=1)
        # x=self.dropout(x)
        # x=F.relu(self.fc2_1(x))
        x=self.dropout(x)
        x=self.fc3(x)

        return x,r1_edge

# Initialize the model
def train(model,train_data,optimizer,crit):
    model.train()
    loss_all = 0
    # get train data
    x=train_data.x.to(device)
    y=train_data.y.to(device)
    size=x.shape[0]

    for i in range(1):
        optimizer.zero_grad()
        output, weight = model(x) 
        # loss=(output-y)*weight
        # loss = crit(loss, torch.zeros_like(loss).to(device))
        loss = crit(output, y)
        # loss+=crit(output[(output-y)<0],y[(output-y)<0])
        # loss-=(crit(torch.clip(output_neg,-1e6,1e6),y))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        size+=x.size()[0]
    return loss_all / size

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

        temp,temp_neg=model(x,test_size)
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
    return loss_all,pred,gt


# # TEST ONLY
# all_mae=[]
# all_median=[]
# all_mape=[]
# for bmidx in ['ab_fixedge']:
#     model=MyModel(x_train.shape[1],2048,  K=5, device=device).to(device)
#     print(f'{model.get_params_count():E}')
#     try:
#         model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
#     except:
#         print('no saved model')
#     test_loss,pred,gt=test(model,train_data, test_data, nn.L1Loss(reduction='sum'))
#     y_est_pred=pred

#     mae=metrics.mean_absolute_error(y_test,y_est_pred)
#     median_ae=metrics.median_absolute_error(y_test,y_est_pred)
#     mape=metrics.mean_absolute_percentage_error(y_test,y_est_pred)
#     print("The MAE on test set: {:.4f}".format(mae))
#     print("The median AE on test set: {:.4f}".format(median_ae))
#     print("The MAPE on test set: {:.4f}".format(mape))
#     all_mae.append(mae)
#     all_median.append(median_ae)
#     all_mape.append(mape)
# print(f'{MODEL_NAME}:{np.mean(all_mae):.2f}$\pm${np.std(all_mae):.0f}')
# print(f'{MODEL_NAME}:{np.mean(all_median):.2f}$\pm${np.std(all_median):.0f}')
# print(f'{MODEL_NAME}:{np.mean(all_mape):.4f}$\pm${np.std(all_mape):.4f}')
# import sys
# sys.exit()
# # END OF TEST,STOP THE PROGRAM



Training=True
# if Training: 
all_loss=[]
# for k in [4,4,4,4,4]:
# for bmidx in ['ab_fixedge']:
for bmidx in [1]:
    for k in [5]:
        print(f'k: {k}')
        # load the model from saved parameters
        model=MyModel(x_train.shape[1], 2048,  K=k, device=device).to(device)
        
        print(f'{model.get_params_count():E}')
        try:
            model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
        except:
            print('no saved model')
        optimizer = torch.optim.Adam(model.parameters(),  lr=1e-2) 
        # initialize the loss function using mean squared error
        crit=nn.MSELoss(reduction='sum')
        # crit=nn.HuberLoss(reduction='sum',delta=1e1)

        history=[]
        val_history=[]
        test_history=[]

        best_model=copy.deepcopy(model.state_dict())
        skip_batch=0
        for idx, data in enumerate(train_data_loader):
            if not Training:
                break
            if skip_batch>0:
                skip_batch-=1
                continue
            x,y=data
            for repeat in range(3):
                # # sample 1000 node from x
                # idx=np.random.choice(x.shape[0],int(x.shape[0]*0.5),replace=False)
                # # sort id from small to large
                # idx=np.sort(idx)
                # x1=x[idx]
                # y1=y[idx]

                train_data_batch=Data(x=x,y=y)     
                model.r1_edge_index=None
                for param_group in optimizer.param_groups:
                    param_group['lr'] =1e-3
                validate_score_non_decrease_count = 0
                val_loss=0
                best_val_loss=1e15

                for epoch in range(200):
                    loss=train(model,train_data_batch, optimizer,crit)
                    if epoch%10==0:
                        val_loss, pred,gt=test(model,train_data_batch, val_data, nn.L1Loss(reduction='sum'))
                        # plot
                        loc=x_val[:,11] #pca
                        plt.figure(figsize=(15,5))
                        # plt.subplot(1,2,1)
                        plt.plot(loc,pred,'o', label='predicted', color='red')
                        plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
                        plt.legend(loc='best')
                        plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {val_loss:.4f}')
                        plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_val_result.png')
                        plt.close()

                        if val_loss<best_val_loss:
                            validate_score_non_decrease_count=0
                            # best_val_loss=val_loss
                            best_val_loss=val_loss
                            best_model=copy.deepcopy(model.state_dict())
                            # torch.save(model.state_dict(), f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt')
                        else:
                            validate_score_non_decrease_count+=1
                            model.r1_edge_index=None
                        if val_loss>best_val_loss and validate_score_non_decrease_count>2:
                            validate_score_non_decrease_count=0
                            print(f'{epoch} val loss increase, reload with best model params and decrease learning rate')
                            model.load_state_dict(best_model)
                            model.r1_edge_index=None
                            # model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
                            # decrase the learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] =param_group['lr']*0.1
                            if optimizer.param_groups[0]['lr']<=1e-5:
                                break
                    if epoch %20==0:
                        model.r1_edge_index=None

                    history.append(loss)
                    val_history.append(val_loss)
                    #      progress monitor:
                    if (epoch+1) % 20 ==0:
                        print(f'{epoch:3d} -- train loss: {loss:.2E}; val loss: {val_loss:.2E}')
                torch.save(model.state_dict(), f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt')

            
        test_loss,pred,gt=test(model,train_data, test_data, nn.L1Loss(reduction='sum'))
        # test_loss,pred,gt=test(model,train_data, val_data, nn.L1Loss(reduction='sum'))
        # find the idx of the top 10 largest error       
        idx=np.argsort(np.abs(pred.reshape(-1)-gt.reshape(-1)))[-3:]
        print(f'test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
        # for id in idx:
        #     print(test_orig_df.iloc[id])
        if Training:
            # save the model
            model.r1_edge_index=None
            torch.save(model.state_dict(), f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt')

            # plot the train loss and validation loss
            plt.figure(figsize=(15,5))
            plt.plot(history[:],label='train loss')
            # plot val loss on different scale
            ax=plt.gca()
            ax2=ax.twinx()
            ax2.plot(val_history[:],label='val loss',color='orange')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.title(f'{MODEL_NAME} train loss {loss:.4E} vs val loss {val_loss:.4E}')
            plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_loss.png')
            plt.close()

            

        all_loss.append(test_loss)

    
        
print(f'{MODEL_NAME}:{np.mean(all_loss)},{np.std(all_loss)}')

        
# Test only
# model=Model(x_train.shape[1], 1024, sample_rate=2, K=3).to(device)
# loss=0
# try:
#     model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}.pt'))
# except:
#     print('Test: no saved model')
# # initialize the loss function using mean squared error
# crit=nn.MSELoss(reduction='sum')
# test_loss,pred,gt=test(model,train_data, test_data, nn.L1Loss(reduction='sum'))
# print(f'test loss: {test_loss:.8E}')
# # find the idx of the top 10 largest error
# idx=np.argsort(np.abs(pred.reshape(-1)-gt.reshape(-1)))[:10]
# print(f'test loss: {test_loss:.8E},\n{idx}, \n{pred[idx].reshape(-1)}, \n{gt[idx].reshape(-1)}')
# for id in idx:
#     print(test_orig_df.iloc[id])
# idx=np.argsort((pred.reshape(-1)-gt.reshape(-1)))[:5]
# print(f'test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
# for id in idx:
#     print(test_orig_df.iloc[id])


# plot
loc=x_test[:,11] #pca
plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
plt.plot(loc,pred,'o', label='predicted', color='red')
plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result.png')

# plot scatter plot
# x_loc=x_test[:,1] #long
# y_loc=x_test[:,2] #lat
# diff=pred-gt
# # get boundary
# boundaries_fp='./uc/final_project_hp/data/geo_export.shp'
# boundaries=gpd.read_file(boundaries_fp)
# # plot boundary on the background
# boundaries.plot(figsize=(15,15),color='white',edgecolor='black')
# # plot the scatter plot, with color representing the error
# plt.figure(figsize=(15,15))

# plt.legend(loc='best')
# plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
# plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result_heatmap.png')

# clean GPU memory
torch.cuda.empty_cache()