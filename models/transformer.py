"""
Copyright 2021 S-Lab
"""
# from IPython import embed
# from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip

import math

# 添加
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

# Periodic Positional Encoding 周期性位置编码
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=300):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
# n_head = 8, max_seq_len=196, period=1 Biased

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    # B = len(n_head)
    # slopes = torch.ones(B)
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = torch.ones(max_seq_len, max_seq_len).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    mask = mask.unsqueeze(0) + alibi   #
    return mask

# Alignment Bias
# memory_mask: :math:`(T, S)`.

# dataset=="motion", dataset=="Text"
def enc_dec_mask(device, dataset, T, S, B):
    # B = len(n_head)
    slopes = torch.ones(B)
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    cla_mask = slopes.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(0)
    return cla_mask.to(device=device)  # (cla_mask==1)



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ScalingConverter(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearBiasedSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = ScalingConverter(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, tgt_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        # print("tgt_B=", B, x.size())
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = self.key(self.norm(x)) #(self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD

        value = self.value(self.norm(x))
        # print("tgt_mask=",tgt_mask.size(), value.size())
        value = (torch.bmm(tgt_mask, value)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearBiasedCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = ScalingConverter(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, memory_mask):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)

        # # B, N, H, HD
        # value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # # B, H, HD, HD
        # attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        # y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)

        # B, N, H, HD
        value = self.value(self.text_norm(xf))
        value = torch.bmm(memory_mask,value) .view(B, H, -1, T) #
        # B, H, HD, HD
        out = torch.einsum('bnhd,bhdl->bnhl', key, value)# 'bnhd,bnhl->bhdl'
        out= torch.transpose(out, 1,2)
        key = torch.transpose(key, 1,3)
        out = torch.einsum('bnhd,bhdl->bnhl', key, out)  # 'bnhd,bnhl->bhdl'
        out = torch.transpose(out, 1, 3)

        attention = torch.einsum('bnhd,bnhl->bhdl', query, out)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        #

        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = ScalingConverter(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
    

class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearBiasedSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearBiasedCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, tgt_mask, memory_mask):
        x = self.sa_block(x, emb, tgt_mask)
        x = self.ca_block(x, xf, emb, memory_mask)
        x = self.ffn(x, emb)
        return x

class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=196,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_text_layers=4,
                 text_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 no_eff=False,
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.period = 15
        # self.batch_size = 1024
        self.PPE = PeriodicPositionalEncoding(self.latent_dim, period=self.period)
        # temporal bias

        self.dataset ="BIVI"  #"vocaset" #
        # Text Transformer
        self.clip, _ = clip.load('ViT-B/32', "cpu")
        if no_clip:
            self.clip.initialize_parameters()
        else:
            set_requires_grad(self.clip, False)
        if text_latent_dim != 512:
            self.text_pre_proj = nn.Linear(512, text_latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, self.time_embed_dim)
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                LinearTemporalDiffusionTransformerDecoderLayer(
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    text_latent_dim=text_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        
    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)   # (B,77)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND

            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def init_biased_mask(self, T, length, period):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                   :n - closest_power_of_2]

        B = len(length)
        max_seq_len=T
        slopes = torch.Tensor(get_slopes(B))
        # B = len(n_head)
        # slopes = torch.ones(B)
        bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
        bias = - torch.flip(bias, dims=[0])
        alibi = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            alibi[i, :i + 1] = bias[-(i + 1):]
        alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
        # mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
        mask = torch.ones(max_seq_len, max_seq_len).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask = mask.unsqueeze(0) + alibi  #
        return mask

    def forward(self, x, timesteps, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        h = self.PPE(h)
        # tgt_mask: :math:`(T, T)`.
        biased_mask = self.init_biased_mask(T=T, length=length, period=self.period)

        # src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)
        tgt_mask = biased_mask[:, :h.shape[1], :h.shape[1]].clone().detach().to(device=x.device)
        memory_mask = enc_dec_mask(h.device, self.dataset, h.shape[1], xf_out.shape[1], B)
        for module in self.temporal_decoder_blocks:
            h = module(h, xf_out, emb, tgt_mask, memory_mask)

        output = self.out(h).view(B, T, -1).contiguous()

        return output




if __name__ == '__main__':
    import numpy as np
    import torch

    import time

    from transformer import  MotionTransformer
    import torch.nn.functional as F
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')


    dim_pose=150
    max_motion_length=300
    num_layers=8
    latent_dim=512
    no_clip= 'store_true'
    no_eff='store_true'
    text = ['die aussichten am mittwoch im nordosten anfangs kräftiger regen sonst ist es wechselhaft im bergland schneeschauer',
                         'und nun die wettervorhersage für morgen sonntag den einundzwanzigsten november']
    text = np.array(text)
    # text = torch.tensor(text).to(device)

    cap_token= torch.IntTensor([[  2,   6, 427,  43, 308,  25, 333, 281, 532, 217,  32, 168,  16,  87,
          25, 473, 914,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
        [  2,   4,   5,   6,   7,   8,   9, 159,  11, 879, 434,   3,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]]).to(device)

    m_lens = torch.IntTensor([180,121]).to(device)
    motion = torch.FloatTensor(2, max_motion_length, dim_pose).to(device)
    timesteps = torch.IntTensor([868,599]).to(device)
    start = time.time()
    model = MotionTransformer(
        input_feats=dim_pose,
        num_frames=max_motion_length,
        num_layers=num_layers,
        latent_dim=latent_dim,
        no_clip=no_clip,
        no_eff=no_eff).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM', (total / 1e6))
    x_start = motion
    B, T = x_start.shape[:2]
    cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(device)

    output = model.forward(motion, timesteps, length=cur_len, text=text)

    torch.cuda.synchronize()
    end = time.time()
    # print('输出数据的维度是:', specs.size())
    print('infer_time', end - start)
    print(output.size())

