import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import GCN
# from torch_geometric.utils import dense_to_sparse

from Selector import FFTSelector


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, cheb_k=2, asymmetric_reg=True, reg_lambda=0.01):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.asymmetric_reg = asymmetric_reg
        self.reg_lambda = reg_lambda
        
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.norm = nn.LayerNorm(dim_out)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
    
    def asymmetric_regularization_loss(self, adj_matrix):
        """
        Compute asymmetric regularization loss
        Args:
            adj_matrix: adjacency matrix [N, N]
        Returns:
            regularization loss
        """
        # Method 1: Penalize asymmetry directly
        asymmetry_loss = torch.norm(adj_matrix - adj_matrix.T, p='fro') ** 2
        
        # Method 2: Add directed edge regularization (optional)
        # Encourage sparsity in one direction while maintaining connectivity
        out_degree = torch.sum(adj_matrix, dim=1)
        in_degree = torch.sum(adj_matrix, dim=0)
        degree_imbalance = torch.norm(out_degree - in_degree, p=2) ** 2
        
        return self.reg_lambda * (asymmetry_loss + 0.1 * degree_imbalance)
    
    def normalize_adjacency(self, adj_matrix, method='sym'):
        """
        Normalize adjacency matrix
        Args:
            adj_matrix: raw adjacency matrix [N, N]
            method: 'sym' for symmetric, 'rw' for random walk, 'asym' for asymmetric
        """
        N = adj_matrix.size(0)
        
        if method == 'sym':
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            rowsum = torch.sum(adj_matrix, dim=1)
            d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_matrix), d_mat_inv_sqrt)
        
        elif method == 'rw':
            # Random walk normalization: D^(-1) A
            rowsum = torch.sum(adj_matrix, dim=1)
            d_inv = torch.pow(rowsum + 1e-8, -1.0)
            d_mat_inv = torch.diag(d_inv)
            return torch.matmul(d_mat_inv, adj_matrix)
        
        elif method == 'asym':
            # Asymmetric normalization for directed graphs
            # Use out-degree for normalization: D_out^(-1) A
            out_degree = torch.sum(adj_matrix, dim=1)
            d_out_inv = torch.pow(out_degree + 1e-8, -1.0)
            d_out_inv_mat = torch.diag(d_out_inv)
            return torch.matmul(d_out_inv_mat, adj_matrix)
        
        else:
            return adj_matrix

    def forward(self, x, supports, embeddings, return_reg_loss=False):
        """
        Forward pass with optional asymmetric regularization
        Args:
            x: input features [B, N, C]
            supports: adjacency matrix [N, N] 
            embeddings: node embeddings [N, D]
            return_reg_loss: whether to return regularization loss
        """
        B, N, D = x.shape
        
        # Apply asymmetric regularization to supports
        if self.asymmetric_reg and self.training:
            reg_loss = self.asymmetric_regularization_loss(supports)
        else:
            reg_loss = torch.tensor(0.0, device=x.device)
        
        # Normalize the adjacency matrix (can choose different methods)
        supports_normalized = self.normalize_adjacency(supports, method='asym')
        
        # Build Chebyshev polynomial support set
        support_set = [torch.eye(N).to(supports.device), supports_normalized]
        
        # Generate higher-order Chebyshev polynomials
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports_normalized, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)  # [cheb_k, N, N]
        
        # Compute adaptive weights and bias
        weights = torch.einsum('nd,dkio->nkio', embeddings, self.weights_pool)  # [N, cheb_k, dim_in, dim_out]
        bias = torch.matmul(embeddings, self.bias_pool)  # [N, dim_out]
        
        # Graph convolution
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # [B, cheb_k, N, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, cheb_k, dim_in]
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, dim_out]
        
        # Layer normalization
        x_gconv = self.norm(x_gconv)
        
        if return_reg_loss:
            return x_gconv, reg_loss
        return x_gconv



# class GCN(nn.Module):
#     def __init__(self, dim_in, dim_out, embed_dim, cheb_k=2):
#         super(GCN, self).__init__()
#         self.cheb_k = cheb_k
#         self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
#         self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

#         self.norm = nn.LayerNorm(dim_out)
    
#     def forward(self, x, supports, embeddings):
#         # x shaped[B, N, C], embeddings shaped [N, D] -> supports shaped [N, N]
#         #output shape [B, N, C]
#         B,N,D = x.shape

#         # print("supports shape:", supports.shape)
#         # print("supports device:", supports.device)

#         support_set = [torch.eye(N).to(supports.device), supports]
#         #default cheb_k = 3
#         for k in range(2, self.cheb_k):
#             support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        
#         supports = torch.stack(support_set, dim=0)

#         # print(supports.shape)
#         # print(embeddings.shape)
#         # print(self.weights_pool.shape)

#         weights = torch.einsum('nd,dkio->nkio', embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
#         bias = torch.matmul(embeddings, self.bias_pool)                       #N, dim_out
#         x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
#         x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
#         x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        
#         x_gconv = self.norm(x_gconv)

#         return x_gconv


class CGCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, N, K):
        super(CGCN, self).__init__()
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.embs_dropout = nn.Dropout(0.1)
        self.layernorm_graph = nn.LayerNorm(N, eps=1e-12)

        self.gcn = GCN(dim_in, dim_out, embed_dim=embed_dim)
        self.weigted_agg = nn.Linear(K,1)

        self.norm = nn.LayerNorm(dim_out)

    
    def forward(self, x, selected_x, selected_relevant_values, selected_indices, node_embeddings, time_embeddings):
        # x: B, T, N, D
        # selected x: B, T, K, N, D
        # selected values: B, T, K
        # node_embeddings: N,d
        # time_embeddings: T,d

        B, T, K, N, D = selected_x.shape

        node_embeddings_expanded = node_embeddings.unsqueeze(1)  # 变为 [N, 1, d]
        time_embeddings_expanded = time_embeddings.unsqueeze(0)  # 变为 [1, T, d]
        full_embeddings = node_embeddings_expanded + time_embeddings_expanded

        # 对于每一个时间步 有NK x NK 的cross graph

        out_set = []
        for t in range(T):
            embeddings_t = self.embs_dropout(self.layernorm(full_embeddings[:,t,:]))
            supports = F.softmax(torch.mm(embeddings_t, embeddings_t.transpose(0, 1)), dim=1)

            relevant_values_t = selected_relevant_values[t,:] # K
            relevant_graph = torch.zeros(size = (N, N, K)).to(x.device)
            for i in range(K):
                diag = torch.ones(size = (N,)).to(x.device)
                diag = diag*relevant_values_t[i]
                diag_graph = torch.diag(diag)*supports
                relevant_graph[:,:,i] = diag_graph

            cross_graph = torch.zeros(size = (N*K, N*K)).to(x.device)
            # layer 0 -> layer 1, ..., layer 0 -> layer L; layer 1 -> layer 2, ..., layer 2 -> layer L
            for i in range(K):
                for j in range(K):
                    if i == j:
                        cross_graph[N * i : N * (i + 1), N * i : N * (i + 1)] = supports + relevant_graph[:,:,j]
                    elif j > i:
                        cross_graph[N * i : N * (i + 1), N * j : N * (j + 1)] = relevant_graph[:,:,j]            
            
            selected_x_t = selected_x[:,t, ...]
            selected_x_t = selected_x_t.permute(0,2,1,3)
            selected_x_t = selected_x_t.reshape(B,K*N,D)

            selected_embed_t = full_embeddings[:,selected_indices[t],:]
            selected_embed_t = selected_embed_t.reshape(N*K,-1)

            # print(cross_graph.shape)
            
            out_t = self.gcn(selected_x_t, cross_graph, selected_embed_t)
            out_set.append(out_t)
        
        out = torch.stack(out_set, dim=0)
        # print(out.shape)

        out = out.reshape(T,B,N,K,D)
        out = out.permute(1,0,2,4,3)

        # out = self.weigted_agg(out).reshape(B,T,N,D)
        out = self.norm(out.mean(-1))
        return out+selected_x.mean(2)


if __name__ == "__main__":
    B, T, N, D = 64, 12, 207, 152
    d = 10
    X = torch.rand(B, T, N, D)
    node = torch.rand(N, d)
    time = torch.rand(T, d)
    K = 3  # 每个时间步找出最相似的 3 个时间步

    selector = FFTSelector(N)
    cgcn = CGCN(D,D,d,N,K)

    selected_values, selected_indices, selected_X = selector(X, K)
    out_ = cgcn(X, selected_X, selected_values, selected_indices, node, time)
    print(out_.shape)