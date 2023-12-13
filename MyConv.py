import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_geometric as tg
class MyConv(MessagePassing):
   
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, include_price=False, reduce='mean', **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_price=include_price

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=True)
        # self.lin_l1 = Linear(out_channels, out_channels, bias=True)
        # if not self.include_price:
        #     self.lin_r = Linear(in_channels[1], out_channels, bias=True)
       
        # self.lin_combine = Linear(out_channels, out_channels, bias=True)
        self.reduce=reduce
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,edge_weight=None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out_u = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        return out_u
        # return self.lin_l(out_u)

    # def message(self, x_i, x_j):
    #     # print('message called.')
    #     tmp = torch.cat([ x_j - x_i], dim=1)
    #     return tmp
    def message(self, x_i, x_j, edge_weight):
        temp=x_j-x_i
        # return torch.tanh(self.lin_l(temp)) 
        return self.lin_l(temp)
        # return temp
       
    def aggregate(self, inputs, index, dim_size=None):
        # print('aggregate called.')
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')


class MyModel(torch.nn.Module):

    def __init__(self, input_size, hs,  K=5, device='cpu') -> None:
        super().__init__()
        self.device=device
        self.K=K
        self.r1_edge_index=None        
        self.r1_edge_attr=None
        self.neg_edge_index=None    
        

        self.fc1=nn.Linear(input_size-1,hs)
     

        self.graph1 = MyConv( hs+1, hs>>1)
        self.fc2=nn.Linear(hs+1,hs)
        t=hs>>1
        self.fc2_1=nn.Linear(hs+t,hs)
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
                weights=weights[weights>0.99]

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
                r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x_wo_price,create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x,create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(torch.cat((x,x_with_price[:,-8:]),dim=1),create_edge_node_size=x.shape[0],start_idx=0,k=k1)
                self.r1_edge_index=r1_edge_index
                self.neg_edge_index=neg_edge_index
                self.r1_edge_attr=r1_edge_attr
              
            else: # test only; always create edges
                r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x_wo_price,create_edge_node_size=test_size,start_idx=test_size,k=k1)
                # r1_edge_index,r1_edge_attr,neg_edge_index=self.create_edges(x,create_edge_node_size=test_size,start_idx=test_size,k=k1)
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

        x_conv =self.graph1((x_i,x_j),r1_edge)
        # batch normalization on the conv output
        # x_conv = F.batch_norm(x_conv, running_mean=None, running_var=None, training=True, momentum=0.1)
        x=F.relu(self.fc2(x_i))
        # x=F.relu(x)+x_conv
        x=torch.cat((x,x_conv),dim=1)
        x=self.dropout(x)
        x=F.relu(self.fc2_1(x))
        x=self.dropout(x)
        x=self.fc3(x)

        return x,r1_edge