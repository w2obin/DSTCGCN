import torch.nn as nn
import torch
from torchinfo import summary

from Selector import FFTSelector
from CGCN import CGCN
from Attention import *


class DSTCGCN(nn.Module):
    def __init__(
        self,
        num_nodes=207,
        embed_dim=10,
        selected_k=3,
        g_layers=1,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.selected_k= selected_k
        self.g_layers=g_layers
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.selectors = nn.ModuleList(
            [
                FFTSelector(self.num_nodes)
                for _ in range(self.g_layers)
            ]
        )

        self.g_net = nn.ModuleList(
            [
                CGCN(self.model_dim, self.model_dim, embed_dim, self.num_nodes, selected_k)
                for _ in range(self.g_layers)
            ]
        )

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(self.in_steps, embed_dim), requires_grad=True)

        # self.mlp = 

        # self.use_norm = True

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        # print(x.shape)
        # print(self.input_proj)
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        # (batch_size, in_steps, num_nodes, model_dim)
        x = torch.cat(features, dim=-1)  

        # atten branch
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim=152)

        # cgcn branch
        # (batch_size, in_steps, num_nodes, model_dim)
        
        x_gcn = torch.cat(features, dim=-1)
        for i in range(self.g_layers):
            selected_values, selected_indices, selected_X = self.selectors[i](x_gcn, self.selected_k)
            x_gcn = self.g_net[i](x_gcn, selected_X, selected_values, selected_indices, self.node_embeddings, self.time_embeddings)
        # (batch_size, in_steps, num_nodes, model_dim)
            

        x = (x+x_gcn)*0.5

        # x = torch.cat([x,x_gcn], dim=-1)
        # x = self.mlp(x)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

if __name__ == "__main__":
    device = 'cuda:1'
    B, T, N, D = 32, 12, 207, 152
    X = torch.rand(B, T, N, D).to(device)
    model = DSTCGCN().to(device)
    o = model(X)
    print(o.shape)
    # summary(model, [B, T, N, D])
