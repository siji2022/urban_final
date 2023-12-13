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

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 160
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv, SAGEConv,GATConv,GraphConv
from torch_geometric.nn import global_mean_pool
import torch_geometric as tg

# fix random seed
torch.manual_seed(0)
np.random.seed(0)

# read the sold data
fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample10k.csv'
a_fn='./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv'
fn_bm='./uc/final_project_hp/data/sold_chicago_5yrs_sample10k.csv'
sold=pd.read_csv(fn)
active_listing=pd.read_csv(a_fn)
sold_bm=pd.read_csv(fn_bm)

x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=data_preprocessing(sold_bm)
x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=data_preprocessing(sold,standardscale=standardscale, labelencoders=labelencoders)
# x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=\
# data_preprocessing_activelisting(active_listing,sold,standardscale=standardscale, labelencoders=labelencoders)
# split x_train and y_train into train and validation
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1, random_state=0)
print(f'x_train shape: {x_train.shape}')

# transform to torch dataset
X_train = torch.from_numpy(x_train).float()
Y_train = torch.from_numpy(y_train).float().unsqueeze(1)
x_val[:,0]=x_train[:,0].min() # set the price to 0 for validation data
X_val = torch.from_numpy(x_val).float()
Y_val = torch.from_numpy(y_val).float().unsqueeze(1)
X_test = torch.from_numpy(x_test).float()
Y_test = torch.from_numpy(y_test).float().unsqueeze(1)


train_data = Data(x=X_train, y=Y_train)
val_data = Data(x=X_val, y=Y_val)
test_data = Data(x=X_test,  y=Y_test)




device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# device = torch.device("cpu")
MODEL_NAME='GNN'




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
        output, output_neg = model(x) 
        loss = crit(output, y)
        loss+=crit(output[(output-y)<0],y[(output-y)<0])
        loss-=(crit(torch.clip(output_neg,0,1e7),y))
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

class Model(torch.nn.Module):

    def __init__(self, input_size, hs, sample_rate=2, K=5,SAGEBias=False) -> None:
        super().__init__()
        self.sample_rate=sample_rate
        self.K=K
        self.r1_edge_index=None        
        self.r2_edge_index=None   
        self.r3_edge_index=None 
        self.neg_edge_index=None    
        
        self.fc1=nn.Linear(input_size-1,hs)
        self.fc2=nn.Linear(hs,hs)
        self.fc3=nn.Linear(hs,hs)
        
        self.conv_p1 = GATConv(1, 1,heads=256, concat=False,add_self_loops=False)
        self.conv_p2 = GATConv(1, 1,heads=256, concat=False,add_self_loops=False)
        self.conv_p3 = GATConv(1, 1,heads=256, concat=False,add_self_loops=False)

        self.graph1 = SAGEConv(hs+3, hs>>1,bias=SAGEBias)
        self.graph2 = SAGEConv(hs+3, hs>>1,bias=SAGEBias)
        self.graph3 = SAGEConv(hs+3, hs>>1,bias=SAGEBias)
        # self.graph1_1 = SAGEConv(hs, hs>>1,bias=SAGEBias)
        # self.graph2_1 = SAGEConv(hs, hs>>1,bias=SAGEBias)
        # self.graph3_1 = SAGEConv(hs, hs>>1,bias=SAGEBias)
        self.conv2_1 = SAGEConv(hs+3, hs, bias=SAGEBias)
        self.conv2_2 = SAGEConv(hs, hs, bias=SAGEBias)


        # self.conv2_2 = SAGEConv(hs, hs)
        # self.conv2_3 = SAGEConv(hs, hs)
        # self.conv2 = SAGEConv(hs*3, hs)
        # self.conv3 = GCNConv(hs, hs)

      
        self.fc4 = nn.Linear((hs>>1)*3, hs)
        self.fc4_n = nn.Linear(hs, hs)
        self.fc5=nn.Linear(hs,hs)
        self.fc6 = nn.Linear(hs, 1)
        self.fc7 = nn.Linear(hs, 1)
        self.dropout = nn.Dropout(0)
        self.counter=0

    def create_edges(self,x,create_edge_node_size=0, start_idx=0, k=5):
        torch.set_grad_enabled(False)
        edge_index_start=[]
        edge_index_end=[]

        neg_edge_index_start=[]
        neg_edge_index_end=[]

        i=0
        # start_idx=0
        # norm x
        x_norm=x/torch.norm(x,dim=1).reshape(-1,1)
        pdist = nn.PairwiseDistance(p=1)
        while i <create_edge_node_size:
            # get the dot product of all other nodes
            # dot_product=torch.matmul(x_norm[i,:],x_norm[:].T)
            # dot_product1=F.cosine_similarity(x[i,:].reshape(1,-1),x[:],dim=1)
            dot_product=pdist(x_norm[i,:].reshape(1,-1),x_norm[:])
            # pick the top 10 neighbors index to add in edge_index
            _,topK=torch.topk(dot_product[start_idx:],k,largest=False)
            # add the edge
            edge_index_start.append((torch.ones(k)*i).to(device))
            edge_index_end.append(topK+start_idx)

            _,topK=torch.topk(dot_product[start_idx:],k,largest=True)
            # add the edge
            neg_edge_index_start.append((torch.ones(k)*i).to(device))
            neg_edge_index_end.append(topK+start_idx)

            i=i+np.random.randint(1,self.sample_rate)
        edge_index=torch.cat((torch.cat(edge_index_start).reshape(-1,1),torch.cat(edge_index_end).reshape(-1,1)),dim=1).long()
        edge_index=edge_index.t().contiguous()
        edge_index=tg.utils.remove_self_loops(edge_index)[0]
        neg_edge_index=torch.cat((torch.cat(neg_edge_index_start).reshape(-1,1),torch.cat(neg_edge_index_end).reshape(-1,1)),dim=1).long()
        neg_edge_index=neg_edge_index.t().contiguous()
        torch.set_grad_enabled(True)
        return edge_index,neg_edge_index


    def forward(self, x,  test_size=0):
        if self.training:
            self.counter+=1
        x_orig=x
        if not self.training:
            SAMPLE_RATE=2
        
        x_price=x_orig[:,0:1]
        x=x_orig[:,1: ]

        x1=F.relu(self.fc1(x))
        x1=self.dropout(x1)
        x2=F.relu(self.fc2(x1))
        x2=self.dropout(x2)
        x3=F.relu(self.fc3(x2))
        # check if model in train mode: test mode will always create edge_index; train mode will create edge_index with probability 0.2
        if not self.training or self.r1_edge_index is None or self.counter%200==0 :
            k1=self.K
            k2=self.K
            k3=self.K
            if self.training:
                r1_edge_index,_=self.create_edges(torch.cat((x_orig[:,1:10],x_orig[:,12:18]),dim=1),create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                r2_edge_index,_=self.create_edges(torch.cat((x_orig[:,1:3],x_orig[:,12:18]),dim=1),create_edge_node_size=x.shape[0],start_idx=0,k=k2)
                r3_edge_index,neg_edge_index=self.create_edges(x3,create_edge_node_size=x.shape[0],start_idx=0,k=k3*2)

                self.r1_edge_index=r1_edge_index
                self.r2_edge_index=r2_edge_index
                self.r3_edge_index=r3_edge_index
                self.neg_edge_index=neg_edge_index
            else: # test only; always create edges
                r1_edge_index,_=self.create_edges(torch.cat((x_orig[:,1:10],x_orig[:,12:18]),dim=1),create_edge_node_size=test_size,start_idx=test_size,k=k1)
                r2_edge_index,_=self.create_edges(torch.cat((x_orig[:,1:3],x_orig[:,12:18]),dim=1),create_edge_node_size=test_size,start_idx=test_size,k=k2)
                r3_edge_index,neg_edge_index=self.create_edges(x3,create_edge_node_size=test_size,start_idx=test_size,k=k3*2)
        else:
            r1_edge_index=self.r1_edge_index
            r2_edge_index=self.r2_edge_index
            r3_edge_index=self.r3_edge_index
            neg_edge_index=self.neg_edge_index
        
        # src, target = r1_edge_index
        # x_price1=tg.utils.scatter(x_price[target], src, dim=0,dim_size=x.shape[0], reduce='mean')
        # src, target = r2_edge_index
        # x_price2=tg.utils.scatter(x_price[target], src, dim=0,dim_size=x.shape[0], reduce='mean')
        # src, target = r3_edge_index
        # x_price3=tg.utils.scatter(x_price[target], src, dim=0,dim_size=x.shape[0], reduce='mean')
        x_price1=F.relu(self.conv_p1(x_price,r1_edge_index))
        x_price2=F.relu(self.conv_p2(x_price,r2_edge_index))
        x_price3=F.relu(self.conv_p3(x_price,r3_edge_index))
        if not self.training:
            x_price1[test_size:]=x_price[test_size:]
            x_price2[test_size:]=x_price[test_size:]
            x_price3[test_size:]=x_price[test_size:]

       
        # if self.counter%100==0 :
        #     print(f'\t before x_price : {torch.mean(x_price[:test_size])},{torch.mean(x_price[test_size:])}, {x_price.max()}')
        #     print(f'\t after  x_price1 : {torch.mean(x_price1[:test_size])},{torch.mean(x_price1[test_size:])}')
        #     print(f'\t after  x_price2 : {torch.mean(x_price2[:test_size])},{torch.mean(x_price2[test_size:])}')

        x1=torch.cat((x3,x_price1,x_price2,x_price3),dim=1)        

        x_11 = F.relu(self.graph1(x1, r3_edge_index))
        # x_11 = F.relu(self.graph1_1(x_11, r3_edge_index))
        x_12 = F.relu(self.graph2(x1, r3_edge_index))
        # x_12 = F.relu(self.graph2_1(x_12, r2_edge_index))
        x_13 = F.relu(self.graph3(x1, r3_edge_index))
        # x_13 = F.relu(self.graph3_1(x_13, r1_edge_index))
        
        x2=torch.cat((x_11,x_12,x_13),dim=1)  
        
        # x = F.relu(self.conv2_2(x, r2_edge_index))
        # x = F.relu(self.conv2_3(x, r3_edge_index))
        # x=torch.cat((x1,x2,x3),dim=1)
        # x = F.relu(self.conv2(x, r3_edge_index))


        x=F.relu(self.fc4(x2))
        x=F.relu(self.fc5(x))
        x=self.fc6(x)

        x_n = F.relu(self.conv2_1(x1, neg_edge_index))
        # x_n = F.relu(self.conv2_2(x_n, neg_edge_index))
        x_n=F.relu(self.fc4_n(x_n))
        x_n=self.fc7(x_n)
        return x, x_n

Training=True
# if Training: 

for b in [ False]:
    for hs in [32]:
        print(f'hs: {hs}, bias: {b}')
        # load the model from saved parameters
        model=Model(x_train.shape[1], hs, sample_rate=15, K=3, SAGEBias=False).to(device)
        # try:
        #     model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}.pt'))
        # except:
        #     print('no saved model')
        optimizer = torch.optim.Adam(model.parameters(),  lr=1e-3) 
        # initialize the loss function using mean squared error
        crit=nn.MSELoss(reduction='sum')

        history=[]
        val_history=[]
        test_history=[]
        best_validate_mae = np.inf
        validate_score_non_decrease_count = 0
        best_epoch=0
        val_loss=0
        for epoch in range(1000):
            loss=train(model,train_data, optimizer,crit)
            if epoch%50==0:
                val_loss=test(model,train_data, val_data, nn.L1Loss(reduction='sum'))[0]
            history.append(loss)
            val_history.append(val_loss)
        #      progress monitor:
            if (epoch+1) % 200 ==0:
                print(f'{epoch:3d} -- train loss: {loss:.2E}; val loss: {val_loss:.2E}')
            # if (epoch+1) ==200:
            #     # decrase the learning rate
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 5e-4


            
        test_loss,pred,gt=test(model,train_data, test_data, nn.L1Loss(reduction='sum'))
        # find the idx of the top 10 largest error
        idx=np.argsort(np.abs(pred.reshape(-1)-gt.reshape(-1)))[-3:]
        print(f'test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
        # save the model
        torch.save(model.state_dict(), f'./uc/final_project_hp/models/{MODEL_NAME}.pt')

        # plot the train loss and validation loss
        plt.figure(figsize=(15,5))
        plt.plot(history[199:],label='train loss')
        # plot val loss on different scale
        ax=plt.gca()
        ax2=ax.twinx()
        ax2.plot(val_history[199:],label='val loss',color='orange')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f'{MODEL_NAME} train loss {loss:.4E} vs val loss {val_loss:.4E}')
        plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_loss_{hs}_{b}.png')
        plt.close()

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
loc=x_test[:,12] #pca
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