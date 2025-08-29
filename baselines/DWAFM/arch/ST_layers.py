import torch

from torch import nn

class Attention_FreMLPs(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.real_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.imag_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dk = hidden_dim // 2
        self.d = 1 / torch.sqrt(torch.tensor(self.dk))

        self.conv_down = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim * 12, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
        )

        self.conv_up = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 12, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv1d(in_channels=hidden_dim * 12, out_channels=hidden_dim * 12, kernel_size=1),
        )

    def self_attention_Node(self, input_data: torch.Tensor) -> torch.Tensor:
        B, T, N, d = input_data.shape
        input_trans = input_data.transpose(1, 2).reshape(B, N, T*d)
        input_trans = self.conv_down(input_trans.transpose(1, 2)).transpose(1, 2)
        query = self.Q(input_trans)
        key = self.K(input_trans).transpose(1, 2)
        value = self.V(input_trans)
        atten_scores = torch.softmax(torch.bmm(query, key) * self.d, dim=-1) # [B, N, N]
        output_data = torch.bmm(atten_scores, value) # [B, N, N] * [B, N, T * d] = [B, N, T * d]
        output_data = self.conv_up(output_data.transpose(1, 2)).transpose(1, 2)
        output_data = output_data.reshape(B, N, T, d).transpose(1, 2) # [B, T, N, d]
        return output_data

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        B, T, N, d = input_data.shape
        ########################Spatial Layer############################
        output_node = self.self_attention_Node(input_data)
        hidden_res1 = output_node + input_data # residual connect
        hidden_norm = self.norm(hidden_res1) # LayerNorm
        ########################Temporal Layer############################
        hidden_ffted = torch.fft.rfft(hidden_norm, dim=1)  # FFT
        hidden_ffted_real = hidden_ffted.real  # real_part
        hidden_ffted_imag = hidden_ffted.imag  # imag_part
        hidden_real = self.real_mlp(hidden_ffted_real) - self.imag_mlp(hidden_ffted_imag) # MLP
        hidden_imag = self.imag_mlp(hidden_ffted_real) + self.real_mlp(hidden_ffted_imag) # MLP
        hidden_complex = torch.complex(hidden_real, hidden_imag) # complex
        hidden_iffted = torch.fft.irfft(hidden_complex, n=T, dim=1)  # IFFT
        hidden = hidden_iffted + input_data  # residual connect
        ###########################################################
        return hidden


