import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTSelector(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feed_forward_dim = feed_forward_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        # self.self_atten = SelfAttentionLayer(self.num_nodes*self.model_dim, self.feed_forward_dim, num_heads, dropout)

        self.q_linear = nn.Linear(self.num_nodes*self.model_dim, self.feed_forward_dim)
        self.k_linear = nn.Linear(self.num_nodes*self.model_dim, self.feed_forward_dim)
    
    def forward(self, X, K):
        B, T, N, D  = X.shape
        x_pack = X.reshape(B,T,-1)
        
        # 标准化时间步
        # X_mean = X.mean(dim=2, keepdim=True)
        # X_std = X.std(dim=2, keepdim=True)
        # X_norm = (X - X_mean) / (X_std + 1e-8)

        q = self.q_linear(x_pack) # B,T,D
        k = self.k_linear(x_pack) # B,T,D

        q_fft = torch.fft.rfft(q)
        k_fft = torch.fft.rfft(k)

        # 计算所有时间步对的互相关系数
        # 通过广播机制计算每对时间步之间的互相关
        X_fft_expanded_i = q_fft.unsqueeze(2)  # (B, T, 1, d)
        X_fft_expanded_j = k_fft.unsqueeze(1)  # (B, 1, T, d)
        
        # 计算频域的共轭乘积，结果形状为 (B, T, T, d)
        cross_corr_fft = X_fft_expanded_i * torch.conj(X_fft_expanded_j)
        
        # 对频谱的共轭乘积应用逆 FFT，得到时域中的互相关
        cross_corr = torch.fft.ifft(cross_corr_fft, dim=-1).real  # 取实部
        
        # 对 B,d 维度求均值，得到最终的互相关系数矩阵，形状为 (T, T)
        corr_matrix = cross_corr.mean(dim=-1).mean(dim=0)
        
        # 忽略对角线元素（自身互相关），将其设为负无穷大
        corr_matrix = corr_matrix.masked_fill(torch.eye(T, device=X.device).bool(), -float('inf'))
        
        # 为每个时间步找出最相似的 K 个时间步
        # topk_values, topk_indices 形状为 (T, K)
        topk_values, topk_indices = torch.topk(corr_matrix, k=K, dim=1)
        
        # print(topk_values[0,:])
        # print(topk_indices[0,:])

        x_expanded = X.unsqueeze(2).expand(B, T, T, N, D)

        # 最相关的在最前面
        # topk_indices_expanded = topk_indices.unsqueeze(0).expand(B,-1, -1)
        # topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, N)
        # topk_indices_expanded = topk_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        # topk_values_from_X = torch.gather(x_expanded, 2, topk_indices_expanded) 
        # return topk_values, topk_indices, topk_values_from_X

        # 时间步最小的在最前面
        # 对 K 按照值从小到大排序
        _, sorted = torch.sort(topk_indices, dim=-1, descending=False)

        sorted_values =  torch.gather(topk_values, 1, sorted)
        sorted_indices = torch.gather(topk_indices, 1, sorted)

        sorted_indices_expanded = sorted_indices.unsqueeze(0).expand(B, -1, -1)
        sorted_indices_expanded = sorted_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, N)
        sorted_indices_expanded = sorted_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        sorted_values_from_X = torch.gather(x_expanded, 2, sorted_indices_expanded) 
        return sorted_values, sorted_indices, sorted_values_from_X



        

if __name__ == "__main__":
    B, T, N, D = 64, 12, 209, 152
    X = torch.rand(B, T, N, D)
    K = 3  # 每个时间步找出最相似的 3 个时间步
    selector = FFTSelector(N)
    selected_values, selected_indices, selected_X = selector(X, K)
    print(selected_values[0,:])
    print(selected_indices[0,:])
    print(selected_X.shape)  # 形状 (B, T, K, N, D)

