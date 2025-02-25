import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_heads
        assert self.head_dim * n_heads == embedding_dim, "embedding_dim must be divisible by n_heads"

        self.proj_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm_factor = 1 / math.sqrt(self.head_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def forward(self, q, k=None, v=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, num_nodes, embedding_dim = q.size()

        # Compute queries, keys, and values
        Q = self.proj_q(q)  # [batch_size, num_nodes, embedding_dim]
        K = self.proj_k(k)  # [batch_size, num_nodes, embedding_dim]
        V = self.proj_v(v)  # [batch_size, num_nodes, embedding_dim]

        # Split into heads
        Q = Q.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, num_nodes, head_dim]
        K = K.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, num_nodes, head_dim]
        V = V.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, num_nodes, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_factor  # [batch_size, n_heads, num_nodes, num_nodes]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [batch_size, n_heads, num_nodes, num_nodes]
            attn_mask = attn_mask.to(dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, num_nodes, num_nodes]
        attn_weights = self.dropout(attn_weights)  # Apply dropout

        # Apply attention weights to V
        context = torch.matmul(attn_weights, V)  # [batch_size, n_heads, num_nodes, head_dim]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embedding_dim)  # [batch_size, num_nodes, embedding_dim]

        # Final linear projection
        out = self.proj_out(context)  # [batch_size, num_nodes, embedding_dim]
        return out, attn_weights

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers=4, dropout=0.1):
        super(GraphAttentionNetwork, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, global_node_features, adjacency_matrix):
        mean_node_features = global_node_features.mean(dim=1, keepdim=True)
        std_node_features = global_node_features.std(dim=1, keepdim=True) + 1e-6
        global_node_features = (global_node_features - mean_node_features) / std_node_features

        x = self.fc_in(global_node_features)  # [batch_size, num_nodes, hidden_dim]

        for i in range(self.num_layers):
            x, _ = self.attention_layers[i](q=x, k=x, v=x, attn_mask=adjacency_matrix)  # [batch_size, num_nodes, hidden_dim]
            x = self.dropout(x)

        out = self.fc_out(x)  # [batch_size, num_nodes, 1]
        return out.squeeze(-1)  # [batch_size, num_nodes]

# Example usage of GraphAttentionNetwork
if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 256
    num_heads = 4
    num_layers = 4  # Number of GAT layers
    dropout = 0.2  # Dropout rate

    model = GraphAttentionNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout)

    batch_size = 32
    num_nodes = 360

    global_node_features = torch.rand(batch_size, num_nodes, input_dim)
    adjacency_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

    adjacency_mask = adjacency_matrix.bool()

    output = model(global_node_features, adjacency_mask)
    print(output.shape)  
