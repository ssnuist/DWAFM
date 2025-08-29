import torch
from torch import nn
from .ST_layers import  Attention_FreMLPs


class attention_ST_A_embed(nn.Module):
    def __init__(self, embed, adj, w):
        super(attention_ST_A_embed, self).__init__()
        self.Q = nn.Parameter(torch.empty(3, embed))
        nn.init.xavier_uniform_(self.Q)
        self.K = nn.Parameter(torch.empty(3, embed))
        nn.init.xavier_uniform_(self.K)
        self.d = 1 / torch.sqrt(torch.tensor(embed))
        self.adj = adj
        self.w = w
    def forward(self, input_data):
        Q = torch.matmul(input_data, self.Q)
        K = torch.matmul(input_data, self.K)
        attention_score = torch.matmul(Q, K.transpose(2, 3)) * self.d * self.adj
        mask = (attention_score == 0)
        attention_score = attention_score.masked_fill(mask, float('-inf'))
        attention_score = torch.softmax(attention_score, dim=-1)
        attention_score = (attention_score + attention_score.transpose(2, 3)) / 2
        A_embed = torch.matmul(attention_score, self.w)
        return A_embed

class DWAFM(nn.Module):


    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.time_of_day_size = model_args["time_of_day_size"]  # 288
        self.day_of_week_size = model_args["day_of_week_size"]  # 7
        self.adj = model_args["adj"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.if_graph = model_args["if_graph"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.input_len, self.num_nodes, self.embed_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_graph:
            self.w = nn.Parameter(torch.empty(self.num_nodes, self.embed_dim))
            nn.init.xavier_uniform_(self.w)
            self.graph_emb = attention_ST_A_embed(self.embed_dim, self.adj.to(self.device), self.w)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.embed_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.embed_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # feature embedding
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        #############################################################################################
        # Spatial Layer and Temporal Layer
        self.hidden_dim = self.embed_dim * (1 + int(self.if_spatial) + int(self.if_graph) +
                                            int(self.if_time_in_day) + int(self.if_day_in_week))

        self.ST_layer = nn.Sequential(
            *[Attention_FreMLPs(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        ##############################################################################################
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim * 1 * self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """Feed forward of DWAFM.
        Args:
            history_data (torch.Tensor): history data with shape [B, T, N, D]
        Returns:
            torch.Tensor: prediction with shape [B, T, N, D]
        """

        input_data = history_data[..., range(self.input_dim)]  # [B, L, N, 3]
        batch_size, time_steps, num_nodes, _ = input_data.shape
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1] * self.time_of_day_size  # [B,L,N]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2] * self.day_of_week_size
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        ##############################Feature Embedding######################################
        time_series_emb = self.time_series_emb_layer(input_data.permute(0, 3, 2, 1)).permute(0, 3, 2,1)

        ##############################Spatial Embedding####################################
        node_emb = []
        if self.if_spatial:
            graph_embed = self.graph_emb(input_data)
            node_emb.append(graph_embed)
            node_emb.append(self.node_emb.expand(batch_size, *self.node_emb.shape))

        ################################Temporal Embedding###################################
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb)
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb)

        ##############################Embedding Layers#######################################
        hidden_emb = torch.cat([time_series_emb] + node_emb + tem_emb, dim=-1)

        ###############################Spatial and Temporal Layers###########################
        hidden= self.ST_layer(hidden_emb)

        #################################Regression Layer####################################
        hidden = hidden.transpose(2, 3)
        hidden = hidden.reshape(batch_size, self.input_len * 1 * self.hidden_dim, num_nodes).unsqueeze(-1)
        prediction = self.regression_layer(hidden)
        return prediction