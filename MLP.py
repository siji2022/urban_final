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
import copy
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
import torch

import torch.nn as nn
import torch.nn.functional as F



# read the sold data
fn='./uc/final_project_hp/data/sold_chicago_5yrs_cleaned.csv'
# fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample10k.csv'
fn='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
fn_bm='./uc/final_project_hp/data/sold_chicago_5yrs_sample_bytypes.csv'
a_fn='./uc/final_project_hp/data/listing_chicago_20231113_cleaned.csv'
sold=pd.read_csv(fn)
active_listing=pd.read_csv(a_fn)
sold_bm=pd.read_csv(fn_bm)

x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df,_=data_preprocessing(sold)
# x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=data_preprocessing(sold,standardscale=standardscale, labelencoders=labelencoders)
# x_train,x_test,y_train,y_test, standardscale, labelencoders,all_columns,test_orig_df=\
# data_preprocessing_activelisting(active_listing,sold,standardscale=standardscale, labelencoders=labelencoders)
# split x_train and y_train into train and validation
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1, random_state=0)
x_train=x_train[:,1:] #remove price
x_val=x_val[:,1:]
x_test=x_test[:,1:]
print(f'x_train shape: {x_train.shape}')



# x_test=x_test[2056:2056+1,:]
# y_test=y_test[2056:2056+1]


# transform to torch dataset
X_train = torch.from_numpy(x_train).float()
Y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_val = torch.from_numpy(x_val).float()
Y_val = torch.from_numpy(y_val).float().unsqueeze(1)
X_test = torch.from_numpy(x_test).float()
Y_test = torch.from_numpy(y_test).float().unsqueeze(1)
train_tensor = torch.utils.data.TensorDataset(X_train, Y_train) 
val_tensor = torch.utils.data.TensorDataset(X_val, Y_val) 
test_tensor = torch.utils.data.TensorDataset(X_test, Y_test) 
train_data_loader = torch.utils.data.DataLoader(train_tensor, batch_size=1024,shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_tensor, batch_size=1024,shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_tensor, batch_size=1024,shuffle=False)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
MODEL_NAME='MLP'
LEN=1



# Initialize the model
def train(model,train_loader,optimizer,crit):
    model.train()
    loss_all = 0
    size=0
    for idx,data in enumerate(train_loader):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        size+=data[0].size()[0]
        optimizer.zero_grad()
        output = model(data[0]) #app_seq[batch,seq,4]  y_seq[batch*4,seq]
        # print(f'{output.shape}, {data[1].shape}')
        loss = crit(output, data[1])
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / size

def test(model,test_loader,crit):
    model.eval()
    loss_all=0
    with torch.no_grad():
        pred=[]
        gt=[]
         #2, 1
        # print(x.shape)
        size=0
        for idx,data in enumerate(test_loader):
            x = data[0].to(device)
            y = data[1].to(device)
            size+=data[0].size()[0]
            temp=model(x)
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
        loss_all=loss_all/size
    return loss_all,pred,gt

class Model(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        channel=2048
        
        self.fc1=nn.Linear(input_size,channel)
        self.fc2=nn.Linear(channel,channel)
        self.fc3=nn.Linear(channel,channel)
        # self.fc4=nn.Linear(channel,channel>>1)
        # self.fc5=nn.Linear(channel>>1,channel>>2)
        self.fc6=nn.Linear(channel,1)
        # dropout layer
        self.dropout = nn.Dropout(p=0)

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x): 
        """forward

        Args:
            x (tensor): batch,seq

        Returns:
            tensor: predited value
        """
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=F.relu(self.fc2(x))
        # x=self.bn1(x)
        x=self.dropout(x)
        x=F.relu(self.fc3(x))

        x=self.dropout(x)
        x=self.fc6(x)
        return x


# TEST ONLY
all_mae=[]
all_median=[]
all_mape=[]
for bmidx in [1,2,3,4,5]:
    model=Model(x_train.shape[1]).to(device)
    print(f'{model.get_params_count():E}')
    try:
        model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
    except:
        print('no saved model')
    test_loss,pred,gt=test(model,test_data_loader, nn.L1Loss(reduction='sum'))
    y_est_pred=pred

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
# END OF TEST,STOP THE PROGRAM




all_loss=[]
for bmidx in [1,2,3,4,5]:
    for sr in [0]:
        # load the model from saved parameters
        model=Model(x_train.shape[1]).to(device)
        
        print(f'{model.get_params_count():E}')
        try:
            model.load_state_dict(torch.load(f'./uc/final_project_hp/models/{MODEL_NAME}_{bmidx}.pt'))
        except:
            print('no saved model')
        optimizer = torch.optim.Adam(model.parameters(),  lr=1e-3) 
        # initialize the loss function using mean squared error
        crit=nn.MSELoss(reduction='sum')

        history=[]
        val_history=[]
        test_history=[]

        validate_score_non_decrease_count = 0
        best_model=model.state_dict().copy()
        val_loss=0
        best_val_loss=1e10
        for epoch in range(400):
            loss=train(model,train_data_loader, optimizer,crit)

            if epoch%5==0:
                val_loss, pred,gt=test(model,val_data_loader, nn.L1Loss(reduction='sum'))
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    best_model=copy.deepcopy(model)
                if val_loss>best_val_loss:
                    print('val loss increase, reload with best model params')
                    model=best_model
                    # decrase the learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] =param_group['lr']*0.2
                    if optimizer.param_groups[0]['lr']<=1e-5:
                        break
                        # best_val_loss=1e10 
                        # optimizer = torch.optim.Adam(model.parameters(),  lr=1e-2)
            history.append(loss)
            val_history.append(val_loss)
            
        #      progress monitor:
            if (epoch+1) % 200 ==0:
                print(f'{epoch:3d} -- train loss: {loss:.2E}; val loss: {val_loss:.2E}')
            # if (epoch+1) ==200:
                

        # plot
        loc=x_val[:,10] #pca
        plt.figure(figsize=(15,5))
        # plt.subplot(1,2,1)
        plt.plot(loc,pred,'o', label='predicted', color='red')
        plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
        plt.legend(loc='best')
        plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, val loss: {val_loss:.4f}')
        plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_val_result.png')
        plt.close()
            
        test_loss,pred,gt=test(model,test_data_loader, nn.L1Loss(reduction='sum'))
        # test_loss,pred,gt=test(model,train_data, val_data, nn.L1Loss(reduction='sum'))
        # find the idx of the top 10 largest error       
        idx=np.argsort(np.abs(pred.reshape(-1)-gt.reshape(-1)))[-3:]
        print(f'test loss: {test_loss:.8E},\n{idx}, \n{pred[idx]}, \n{gt[idx]}')
        # for id in idx:
        #     print(test_orig_df.iloc[id])
        # save the model
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
# plot the train loss and validation loss
# plt.figure(figsize=(15,5))
# plt.plot(history,label='train loss')
# plt.plot(val_history,label='val loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.title(f'{MODEL_NAME} train loss vs val loss')
# plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_loss.png')
# plt.close()

# plot
loc=x_test[:,10] #pca
plt.figure(figsize=(15,5))
# plt.subplot(1,2,1)
plt.plot(loc,pred,'o', label='predicted', color='red')
plt.plot(loc,gt, 'o',label='real', color='blue',alpha=0.5)
plt.legend(loc='best')
plt.title(f'{MODEL_NAME},train loss:{loss:.4f}, test loss: {test_loss:.4f}')
plt.savefig(f'./uc/final_project_hp/figures/{MODEL_NAME}_scatter_test_result.png')