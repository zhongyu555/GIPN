import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.utils import get_root_logger
from timm.models.layers import LayerNorm2d
from mmcv.runner import load_checkpoint
import math
from collections import OrderedDict
import numpy as np
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_

# import sys
# sys.path.append('/path/to/your/package')
from models.bra_legacy import BiLevelRoutingAttention
from models._common import Attention, AttentionLePE, DWConv
from models.Transformer import Transformer

import warnings
warnings.filterwarnings("ignore")

class Conv_GIP(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv_GIP, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.up2x_3_1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.up2x_3_2 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False), # H ,W
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))

    def forward(self, feats):

        feats[0] = self.conv1_1(feats[0])
        feats[1] = self.conv1_2(feats[1])
        feats[2] = self.conv3_1(feats[2])
        feats[3] = self.conv3_2(feats[3])
        feats[4] = self.up2x_3_1(feats[4])
        feats[5] = self.up2x_3_2(feats[5])

        return feats

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Projection(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(Projection, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input
        self.phi = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_phi = nn.BatchNorm2d(dim)
        self.theta = nn.Conv2d(dim, node_num, kernel_size=1, bias=False)
        self.bn_theta = nn.BatchNorm2d(node_num)


    def forward(self, x):
        B, C, H, W = x.size()
        phi = self.phi(x) # B,C,H,W
        phi = self.bn_phi(phi)
        phi = phi.view(B,C,-1).contiguous() # B,C,(HW)
        
        theta = self.theta(x) # B,N,H,W
        theta = self.bn_theta(theta)
        theta = theta.view(B,self.node_num,-1).contiguous() # B,N,(HW)
        nodes = torch.matmul(phi, theta.permute(0, 2, 1).contiguous())  # [B,C,(HW)] * [B,(HW),N] = [B,C,N] ---> [B,N,C]
        return nodes, theta  # nodes [B,N,C],  theta [B,N_node,H,W]


class TransformerLayer(nn.Module):
    def __init__(self, dim, loop):
        super(TransformerLayer, self).__init__()
        self.gt1 = MultiHeadAttentionLayer(hid_dim=dim, n_heads=8, dropout=0.1)
        self.gt2 = MultiHeadAttentionLayer(hid_dim=dim, n_heads=8, dropout=0.1)
        self.gts = [self.gt1, self.gt2]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gts = self.gts[0:loop]

    def forward(self, x):
        for gt in self.gts:
            x = gt(x) # b x c x k
        return x


class GraphTransformer(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.gt = TransformerLayer(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    #graph0: edge, graph1/2: region, assign:edge
    def forward(self, query, key, value, assign):
        # Hierarchical Graph Interaction
        m = self.corr_matrix(query, key, value)
        interacted_graph = query + m  # 为V_i_inter_hat
        
        # Transformer Learning 
        enhanced_graph = self.gt(interacted_graph)  # 这里面进行的拉普拉斯操作和transformer操作
        enhanced_feat = enhanced_graph.bmm(assign) # reprojection
        enhanced_feat = self.conv(enhanced_feat.unsqueeze(3)).squeeze(3)  # 从图映射为空域
        return enhanced_feat

    def corr_matrix(self, query, key, value):
        assign = query.permute(0, 2, 1).contiguous().bmm(key)
        assign = F.softmax(assign, dim=-1) #normalize region-node  assign为S_i->i+1
        m = assign.bmm(value.permute(0, 2, 1).contiguous())  # ".bmm"为批量矩阵乘法，这里是乘(QK)和V, m为V_i_inter_hat没加V_i_hat之前
        m = m.permute(0, 2, 1).contiguous()
        return m


class GIP(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=768, num_clusters=8, dropout=0.1):
        super(GIP, self).__init__()

        self.dim = dim

        self.proj_1 = Projection(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.proj_2 = Projection(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.conv1_1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1_2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        self.conv2_1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv2_2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        
        self.conv3 = nn.Sequential(BasicConv2d(self.dim*2, self.dim, BatchNorm, kernel_size=1, padding=0))
        
        self.gt1 = GraphTransformer(self.dim, BatchNorm, dropout) 
        self.gt2 = GraphTransformer(self.dim, BatchNorm, dropout) 

    def forward(self, feat1, feat2):
        # project features to graphs
        graph_2, assign_2 = self.proj_2(feat2) # F3  这里assign_2是在从空域投影到图，保留的经过theta的特征，用于从图再反映射回空域
        graph_1, assign_1 = self.proj_1(feat1) # F4

        q_graph1 = self.conv1_1(graph_1.unsqueeze(3)).squeeze(3) # Q
        k_graph1 = self.conv1_2(graph_1.unsqueeze(3)).squeeze(3) # K

        # Hierarchical Graph Interaction && Transformer Learning and Reprojection
        q_graph2 = self.conv2_1(graph_2.unsqueeze(3)).squeeze(3) # Q 1*1卷积
        k_graph2 = self.conv2_2(graph_2.unsqueeze(3)).squeeze(3) # K 1*1卷积
        
        mutual_V = torch.cat((graph_2, graph_1), dim=1)
        mutual_V_new = self.conv3(mutual_V.unsqueeze(3)).squeeze(3) # V

        enhanced_feat1 = self.gt1(q_graph1, k_graph2, mutual_V_new, assign_1)
        feat1 = feat1 + enhanced_feat1.view(feat1.size()).contiguous()
        
        enhanced_feat2 = self.gt2(q_graph2, k_graph1, mutual_V_new, assign_2)
        feat2 = feat2 + enhanced_feat2.view(feat2.size()).contiguous()

        return feat1, feat2


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, BatchNorm=nn.BatchNorm2d):
        super(MultiHeadAttentionLayer, self).__init__()
        self.nodes = 8
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.node_conv = nn.Conv1d(hid_dim, hid_dim, 1, bias=True)
        self.eigen_vec_conv = nn.Conv1d(self.nodes, hid_dim, 1, bias=True)

        self.fc_q = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        self.fc_k = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        self.fc_v = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        
        self.fc_o = nn.Conv1d(self.nodes, self.nodes, 1, bias=False)
        
        self.w_1 = nn.Conv1d(hid_dim, hid_dim*4, 1) # position-wise
        self.w_2 = nn.Conv1d(hid_dim*4, hid_dim, 1) # position-wise
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(dropout)
        self.a4 = nn.ReLU(inplace=True)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def get_lap_vec(self, graph):
        device = graph.device
        batch = graph.shape[0]
        graph_t = graph.permute(0, 2, 1).contiguous() # transpose
        dense_adj = torch.matmul(graph_t, graph)
        dense_adj = self.a4(dense_adj)
        in_degree = dense_adj.sum(dim=1).view(batch, -1)
        dense_adj = dense_adj.detach().cpu().float().numpy()
        in_degree = in_degree.detach().cpu().float().numpy()
        number_of_nodes = self.nodes

        # Laplacian
        A = dense_adj
        N = []
        for i in range(0,batch):
            N1 = np.diag(in_degree[i].clip(1) ** -0.5)
            N.append(N1)
        N = np.array(N)
        L = np.eye(number_of_nodes) - N @ A @ N

        # (sorted) eigenvectors with numpy
        try:
            EigVal, EigVec = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            for i in L.shape[0]:
                filename = "laplacian_{}.txt".format(i)
                np.savetxt(filename, L[i])

        
        eigvec = torch.from_numpy(EigVec).float().to(device) 
        eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float().to(device) 
        return eigvec, eigval 
        
    
    def forward(self, qkv):
        
        batch_size = qkv.shape[0]
        device = qkv.device
        scale = self.scale.to(device)
        eigvec, eigval = self.get_lap_vec(qkv)
        qkv = self.node_conv(qkv)
        eigvec = self.eigen_vec_conv(eigvec)
        qkv = qkv + eigvec
        residual = qkv
        
        Q = self.fc_q(qkv)
        K = self.fc_k(qkv)
        V = self.fc_v(qkv)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        
        attention = self.dropout(torch.softmax(energy, dim = -1))
                
        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.dropout(self.fc_o(x))
        residual = residual.permute(0, 2, 1).contiguous()
        x = x + residual
        
        residual = x = self.layer_norm(x).permute(0, 2, 1).contiguous()
        #x = [batch size, query len, hid dim]

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1).contiguous()
        return x



class Model(nn.Module):
    def __init__(self, ckpt=None, img_size=512, num_clusters=8):  # change img size to 512
        super(Model, self).__init__()
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location='cpu') # pretrain pth path
            msg = self.encoder.load_state_dict({k.replace('backbone.',''):v for k,v in ckpt['model'].items()}, strict=False)
            print(msg)

        self.img_size = img_size
        self.vit_chs = 768
        self.num_clusters = num_clusters

        self.conv_gip = Conv_GIP(in_chs=768, out_chs=384)

        self.gip_0 = GIP(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=self.num_clusters, dropout=0.1)
        self.gip_1 = GIP(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=self.num_clusters, dropout=0.1)
        self.gip_2 = GIP(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=self.num_clusters, dropout=0.1)

    def forward(self, x_list):
        x = x_list  #
        feature = []
        out1, out2 = self.gip_0(x[3], x[2])
        feature.append(out1)
        feature.append(out2)

        out1, out2 = self.gip_1(x[2], x[1])
        feature.append(out1)
        feature.append(out2)

        out1, out2 = self.gip_2(x[1], x[0])
        feature.append(out1)
        feature.append(out2)

        feature_new = self.conv_gip(feature)

        return feature_new

