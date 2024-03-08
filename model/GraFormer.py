from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model.blocks.ChebConv import ChebConv, _ResChebGC
from model.blocks.refine import refine


edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                    [0, 4], [4, 5], [5, 6],
                    [0, 7], [7, 8], [8, 9], [9,10],
                    [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16]], dtype=torch.long)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16],
                          [0, 17], [17, 18], [18, 19], [19, 20]], dtype=torch.long)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size, 3 * size, bias=True)
        )

    def forward(self, x, sublayer,c):
        shift, scale, gate= self.adaLN_modulation(c).chunk(3, dim=2)        
        return x + gate * self.dropout(sublayer(modulate(self.norm(x), shift, scale)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask,c):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask),c)
        return self.sublayer[1](x, self.feed_forward,c)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True) # 生成一个对角矩阵
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1
    
    
class FinalLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.gconv = ChebConv(in_c=dim, out_c=3, K=2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x,adj, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm(x), shift, scale)
        x = self.gconv(x,adj)
        return x


class GraFormer(nn.Module):
    def __init__(self, hid_dim=128, coords_dim=(5, 3), num_layers=4,
                 n_head=4,  dropout=0.1, n_pts=17, is_train=True):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.is_train = is_train
        
        self.adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()


        _gconv_input = ChebConv(in_c=6, out_c=hid_dim, K=2)
        
        _gconv_cond = ChebConv(in_c=2, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hid_dim),
            nn.Linear(hid_dim, hid_dim*2),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(hid_dim*2, hid_dim),
        )
        

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        
        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_cond = _gconv_cond
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)
        self.fusion = refine(3)

    def forward(self, x_2d, x_3d,x_pred, t, cam=None):
        if self.is_train:
            x = torch.cat((x_pred, x_3d), dim=-1)
            B, F, J, _ = x.shape
            x = x.reshape(-1, J, 6)
            BF = x.shape[0]
            x = self.gconv_input(x, self.adj)
            _, J, C = x.shape
            
            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1,F,J,1)
            x_2d = x_2d.reshape(-1, J, 2)
            cond_embed = self.gconv_cond(x_2d,self.adj).reshape(B, F, J, C)
            c = time_embed + cond_embed    
            c = c.reshape(BF,J,C)
        else:
            x_2d_h = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            
            if x_pred.shape != x_3d.shape:
                x_pred_h = x_pred[:,None].repeat(1,x_3d.shape[1],1,1,1)
            else:
                x_pred_h = x_pred
            x = torch.cat((x_pred_h, x_3d), dim=-1)
            B, H, F, J, _ = x.shape
            x = x.reshape(-1, J, 6)
            BHF = x.shape[0]
            
            x = self.gconv_input(x, self.adj)
            _, J, C = x.shape
            
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, H, F, J, 1)
            x_2d_h = x_2d_h.reshape(-1, J, 2)
            cond_embed = self.gconv_cond(x_2d_h,self.adj).reshape(B, H, F, J, C)
            c = time_embed + cond_embed
            c = c.reshape(BHF,J,C)
            
            
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, self.src_mask,c)
            x = self.gconv_layers[i](x)

        x = self.gconv_output(x, self.adj)
        if self.is_train:
            x = x.reshape(B, F, J, -1)
            x_pred = x_pred.reshape(B, F, J, -1)
            x = self.fusion(x,x_pred)
            return x
        else:
            x = x.reshape(B*H, F, J, -1)
            if x_pred.shape != x_3d.shape:
                x_pred = x_pred.reshape(B, F, J, -1).unsqueeze(1).repeat(1, H, 1, 1, 1).reshape(B*H, F, J, -1)
            else:
                x_pred = x_pred.reshape(B*H, F, J, -1)
            
            x = self.fusion(x,x_pred).reshape(B,H, F, J, -1)
            
            return x


