import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import GCN
# from torch_geometric.utils import dense_to_sparse
import torch.sparse as sparse
from Selector import FFTSelector

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, cheb_k=2):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.norm = nn.LayerNorm(dim_out)
    
    def _sparse_cheb_polynomials(self, adj_sparse, k):
        """Generate Chebyshev polynomials using sparse matrices"""
        device = adj_sparse.device
        N = adj_sparse.size(0)
        
        # T0 = I (identity matrix as sparse)
        eye_indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
        eye_values = torch.ones(N, device=device)
        T0 = torch.sparse_coo_tensor(eye_indices, eye_values, (N, N), device=device)
        
        if k == 1:
            return [T0]
        
        # T1 = L (normalized adjacency matrix)
        T1 = adj_sparse
        support_set = [T0, T1]
        
        # Generate higher order polynomials: T_k = 2*L*T_{k-1} - T_{k-2}
        for i in range(2, k):
            # 2*L*T_{k-1}
            temp = torch.sparse.mm(T1, support_set[-1]) * 2
            # T_k = 2*L*T_{k-1} - T_{k-2}
            T_new = temp - support_set[-2]
            support_set.append(T_new)
        
        return support_set
    
    def _sparse_mm_batch(self, sparse_matrices, dense_tensor):
        """Efficient batch sparse matrix multiplication"""
        # sparse_matrices: list of [N, N] sparse tensors
        # dense_tensor: [B, N, C]
        B, N, C = dense_tensor.shape
        k = len(sparse_matrices)
        
        results = []
        for sparse_mat in sparse_matrices:
            # [N, N] @ [B, N, C] -> [B, N, C]
            result = torch.stack([torch.sparse.mm(sparse_mat, dense_tensor[b]) for b in range(B)])
            results.append(result)
        
        return torch.stack(results, dim=1)  # [B, k, N, C]
    
    def forward(self, x, adj_sparse, embeddings):
        """
        Args:
            x: [B, N, C] input features
            adj_sparse: sparse adjacency matrix [N, N]
            embeddings: [N, D] node embeddings
        Returns:
            x_gconv: [B, N, dim_out] convolved features
        """
        B, N, C = x.shape
        
        # Generate Chebyshev polynomials as sparse matrices
        support_set = self._sparse_cheb_polynomials(adj_sparse, self.cheb_k)
        
        # Compute adaptive weights and bias
        weights = torch.einsum('nd,dkio->nkio', embeddings, self.weights_pool)  # [N, k, dim_in, dim_out]
        bias = torch.matmul(embeddings, self.bias_pool)  # [N, dim_out]
        
        # Apply sparse convolution
        x_g = self._sparse_mm_batch(support_set, x)  # [B, k, N, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, k, dim_in]
        
        # Weighted aggregation
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, dim_out]
        
        x_gconv = self.norm(x_gconv)
        return x_gconv


class CGCN(nn.Module):
    def __init__(self, dim_in, dim_out, embed_dim, N, K, sparsity_threshold=0.05):
        super(CGCN, self).__init__()
        self.N = N
        self.K = K
        self.sparsity_threshold = sparsity_threshold  # Lower threshold for larger graphs
        
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.embs_dropout = nn.Dropout(0.1)
        self.layernorm_graph = nn.LayerNorm(N, eps=1e-12)
        
        self.gcn = GCN(dim_in, dim_out, embed_dim=embed_dim)
        self.weighted_agg = nn.Linear(K, 1)
        self.norm = nn.LayerNorm(dim_out)
    
    def _create_sparse_adjacency(self, embeddings):
        """Create sparse adjacency matrix from embeddings"""
        # Compute attention scores
        scores = torch.mm(embeddings, embeddings.transpose(0, 1))
        attention = F.softmax(scores, dim=1)
        
        # Apply sparsity threshold
        mask = attention > self.sparsity_threshold
        
        # Get sparse indices and values
        indices = mask.nonzero(as_tuple=False).t()
        values = attention[mask]
        
        N = embeddings.size(0)
        sparse_adj = torch.sparse_coo_tensor(indices, values, (N, N), device=embeddings.device)
        
        return sparse_adj
    
    def _create_sparse_cross_graph(self, relevant_values, base_adj_sparse, device):
        """Create sparse cross-layer graph efficiently"""
        N, K = self.N, self.K
        total_nodes = N * K
        
        # Pre-allocate lists for indices and values
        all_indices = []
        all_values = []
        
        # Convert base adjacency to dense for diagonal operations (only when needed)
        base_adj_dense = base_adj_sparse.to_dense()
        
        for i in range(K):
            for j in range(K):
                row_offset = N * i
                col_offset = N * j
                
                if i == j:
                    # Intra-layer connections: base adjacency + relevant values
                    diag_value = relevant_values[j].item()  # Convert tensor to scalar
                    enhanced_adj = base_adj_dense + torch.diag(torch.full((N,), diag_value, device=device))
                    
                    # Find non-zero elements
                    nonzero_mask = enhanced_adj != 0
                    rows, cols = nonzero_mask.nonzero(as_tuple=True)
                    
                    if len(rows) > 0:
                        # Adjust indices for block structure
                        global_rows = rows + row_offset
                        global_cols = cols + col_offset
                        values = enhanced_adj[rows, cols]
                        
                        all_indices.append(torch.stack([global_rows, global_cols]))
                        all_values.append(values)
                    
                elif j > i:
                    # Inter-layer connections: only relevant value on diagonal
                    diag_value = relevant_values[j].item()  # Convert tensor to scalar
                    
                    # Create diagonal connections with relevant values
                    if abs(diag_value) > 1e-8:  # Filter out very small values
                        rows = torch.arange(N, device=device)
                        cols = torch.arange(N, device=device)
                        
                        # Scale base adjacency diagonal by relevant value
                        diagonal_mask = base_adj_dense.diagonal() != 0
                        if diagonal_mask.any():
                            filtered_rows = rows[diagonal_mask]
                            filtered_cols = cols[diagonal_mask]
                            values = base_adj_dense.diagonal()[diagonal_mask] * diag_value
                        else:
                            # If no diagonal elements in base adjacency, create identity-like connections
                            filtered_rows = rows
                            filtered_cols = cols
                            values = torch.full((N,), diag_value, device=device)
                        
                        if len(filtered_rows) > 0:
                            global_rows = filtered_rows + row_offset
                            global_cols = filtered_cols + col_offset
                            
                            all_indices.append(torch.stack([global_rows, global_cols]))
                            all_values.append(values)
                
                # Skip lower triangular part (j < i)
        
        if all_indices:
            indices = torch.cat(all_indices, dim=1)
            values = torch.cat(all_values)
            
            sparse_cross_graph = torch.sparse_coo_tensor(
                indices, values, (total_nodes, total_nodes), device=device
            )
        else:
            # Create empty sparse matrix
            indices = torch.zeros((2, 0), dtype=torch.long, device=device)
            values = torch.zeros(0, device=device)
            sparse_cross_graph = torch.sparse_coo_tensor(
                indices, values, (total_nodes, total_nodes), device=device
            )
        
        return sparse_cross_graph
    
    def forward(self, x, selected_x, selected_relevant_values, selected_indices, node_embeddings, time_embeddings):
        """
        Args:
            x: [B, T, N, D] original input
            selected_x: [B, T, K, N, D] selected features
            selected_relevant_values: [T, K] or [B, T, K] relevance values
            selected_indices: [T, K] selected time indices
            node_embeddings: [N, d] node embeddings
            time_embeddings: [T, d] time embeddings
        """
        B, T, K, N, D = selected_x.shape
        
        # Handle different dimensions of selected_relevant_values
        if selected_relevant_values.dim() == 2:
            # Shape: [T, K]
            relevant_values = selected_relevant_values
        elif selected_relevant_values.dim() == 3:
            # Shape: [B, T, K] - average across batches
            relevant_values = selected_relevant_values.mean(0)  # [T, K]
        else:
            raise ValueError(f"Unexpected dimension for selected_relevant_values: {selected_relevant_values.shape}")
        
        # Combine node and time embeddings
        node_embeddings_expanded = node_embeddings.unsqueeze(1)  # [N, 1, d]
        time_embeddings_expanded = time_embeddings.unsqueeze(0)  # [1, T, d]
        full_embeddings = node_embeddings_expanded + time_embeddings_expanded  # [N, T, d]
        
        out_set = []
        
        for t in range(T):
            embeddings_t = self.embs_dropout(self.layernorm(full_embeddings[:, t, :]))  # [N, d]
            
            # Create sparse base adjacency matrix
            base_adj_sparse = self._create_sparse_adjacency(embeddings_t)
            
            # Get relevant values for this time step
            relevant_values_t = relevant_values[t, :]  # [K]
            
            # Create sparse cross-graph
            cross_graph_sparse = self._create_sparse_cross_graph(
                relevant_values_t, base_adj_sparse, x.device
            )
            
            # Prepare input for GCN
            selected_x_t = selected_x[:, t, ...].permute(0, 2, 1, 3)  # [B, N, K, D]
            selected_x_t = selected_x_t.reshape(B, K * N, D)  # [B, K*N, D]
            
            # Prepare embeddings
            selected_embed_t = full_embeddings[:, selected_indices[t], :]  # [N, K, d]
            selected_embed_t = selected_embed_t.reshape(N * K, -1)  # [N*K, d]
            
            # Apply sparse GCN
            out_t = self.gcn(selected_x_t, cross_graph_sparse, selected_embed_t)
            out_set.append(out_t)
        
        # Stack and reshape outputs
        out = torch.stack(out_set, dim=0)  # [T, B, N*K, D]
        out = out.reshape(T, B, N, K, D)
        out = out.permute(1, 0, 2, 4, 3)  # [B, T, N, D, K]
        
        # Aggregate and normalize
        out = self.norm(out.mean(-1))  # [B, T, N, D]
        
        # Add residual connection
        residual = selected_x.mean(2)  # [B, T, N, D] - average across K
        
        return out + residual


# Memory-efficient utility functions
# class SparseGraphUtils:
#     @staticmethod
#     def sparse_softmax(sparse_tensor, dim=1):
#         """Apply softmax to sparse tensor along specified dimension"""
#         indices = sparse_tensor.indices()
#         values = sparse_tensor.values()
#         shape = sparse_tensor.shape
        
#         # Group by rows (assuming dim=1)
#         if dim == 1:
#             row_indices = indices[0]
#             unique_rows, inverse_indices = torch.unique(row_indices, return_inverse=True)
            
#             # Apply softmax row-wise
#             softmax_values = torch.zeros_like(values)
#             for i, row_idx in enumerate(unique_rows):
#                 mask = inverse_indices == i
#                 row_values = values[mask]
#                 softmax_values[mask] = F.softmax(row_values, dim=0)
            
#             return torch.sparse_coo_tensor(indices, softmax_values, shape)
        
#         return sparse_tensor
    
#     @staticmethod
#     def sparse_threshold(sparse_tensor, threshold):
#         """Apply threshold to sparse tensor values"""
#         indices = sparse_tensor.indices()
#         values = sparse_tensor.values()
#         shape = sparse_tensor.shape
        
#         mask = values > threshold
#         filtered_indices = indices[:, mask]
#         filtered_values = values[mask]
        
#         return torch.sparse_coo_tensor(filtered_indices, filtered_values, shape)



# Example usage and testing
if __name__ == "__main__":
    # Your actual experiment parameters
    B, T, N, D = 64, 12, 207, 152
    d = 10  # embedding dimension
    K = 3   # number of similar time steps
    
    # Create test data matching your setup
    X = torch.rand(B, T, N, D)
    node = torch.rand(N, d)
    time = torch.rand(T, d)
    
    # Simulate FFTSelector outputs (you'll replace this with actual selector)
    selected_X = torch.rand(B, T, K, N, D)  # Selected features
    selected_values = torch.rand(T, K)      # Relevance values
    selected_indices = torch.randint(0, T, (T, K))  # Selected time indices
    
    # Initialize sparse CGCN with your parameters
    cgcn = CGCN(
        dim_in=D,      # 152
        dim_out=D,     # 152
        embed_dim=d,   # 10
        N=N,           # 207
        K=K,           # 3
        sparsity_threshold=0.05  # Adjusted for larger graph
    )
    selector = FFTSelector(N)
    selected_values, selected_indices, selected_X = selector(X, K)
    out_ = cgcn(X, selected_X, selected_values, selected_indices, node, time)
    print(out_.shape)
