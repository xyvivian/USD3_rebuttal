import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
    
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb




class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        # attention operation
        x_skip = x
        
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim,is_zero=True, std=0.08):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        if is_zero:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
        else:
            init.normal_(self.linear.weight, mean=0.0, std=std)
            init.normal_(self.linear.bias, mean=0.0, std=std)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_() 
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
    


class DDitTransformer(nn.Module):
    def __init__(self, config):
        super(DDitTransformer,self).__init__()
        self.config = config
        self.absorb = config.diffusion.noise_type == "absorb"
        vocab_size = config.diffusion.num_classes + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.length)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.length, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.length,is_zero=False)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)
        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
        return x #.type(torch.float32)
    
    
    
#================================= Regular Sequence Transformer Encoder ===============================================#
def get_ffn_unit(in_dim,
                 ffn_unit="linear",
                 hidden_dim = None,
                 out_dim = None,
                 activation = F.silu,
                 ffn_dropout = 0.0,
                 bias = True):
    if ffn_unit == "glu":
        return GatedLinearFFN(in_dim, hidden_dim,out_dim,activation, ffn_dropout,bias)
    elif ffn_unit == "linear":
        return LinearFFN(in_dim, hidden_dim,out_dim,activation,ffn_dropout,bias)

class GatedLinearFFN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim =None,
                 out_dim = None,
                 activation =F.silu,
                 ffn_dropout = 0.0,
                 bias = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (4*in_dim)
        if out_dim is None:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim,hidden_dim,bias=bias)
        self.gated_fc = nn.Linear(in_dim,hidden_dim,bias=bias)
        self.fc2 = nn.Linear(hidden_dim,out_dim,bias=bias)
        self.dropout=nn.Dropout(ffn_dropout)
        self.activation = activation
    
    def forward(self,X):
        out = self.gated_fc(X)*self.activation(self.fc1(X))
        return self.dropout(self.fc2(self.dropout(out)))
    
    
class LinearFFN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim = None,
                 out_dim = None,
                 activation=F.silu,
                 ffn_dropout = 0.0,
                 bias = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (4*in_dim)
        if out_dim is None:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim,hidden_dim,bias=bias)
        self.fc2 = nn.Linear(hidden_dim,out_dim,bias=bias)
        self.dropout=nn.Dropout(ffn_dropout)
        self.activation = activation   
        
    def forward(self,X):
        return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(X)))))     
        
    
    
class AttentionLayer(nn.Module):
    def __init__(self,d_model,n_heads,attention_dropout = 0.1):
        super().__init__()
        d_keys = (d_model // n_heads)
        d_values = (d_model // n_heads)
        self.dropout = nn.Dropout(attention_dropout)
        
        self.query_proj = nn.Linear(d_model, d_keys* n_heads)
        self.key_proj = nn.Linear(d_model, d_keys * n_heads)
        self.value_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values):
        B,L,_ = queries.shape
        _,S,_ = keys.shape
        H = self.n_heads
        queries= self.query_proj(queries).view(B,L,H,-1)
        E = queries.shape[-1]
        keys = self.key_proj(keys).view(B,S,H,-1)
        values = self.value_proj(values).view(B,S,H,-1)
    
        scale = 1/ math.sqrt(E)
        scores = torch.einsum('blhe,bshe -> bhls',queries,keys)
        A = self.dropout(torch.softmax(scale*scores),dim=-1)
        V = torch.einsum('bhls,bshd->blhd',A,values)
        V = V.view(B,L,-1)
        return self.out_proj(V)
    
    
class  EncoderLayer(nn.Module):
    def __init__(self,attention,feedforward, d_model,temb_dim,dropout,prenorm=False):
        super().__init__()
        self.attenion = attention
        self.feedforward = feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.prenorm = prenorm
        self.dropout = nn.Dropout(dropout)
        self.film_from_temb = nn.Linear(temb_dim, 2*d_model)
    
    def forward(self,x,temb):
        K = x.shape[-1]
        film_params = self.film_from_temb(temb)
        if self.prenorm:
            new_x = self.attention(self.norm1(x),self.norm1(x),self.norm1(x))
            x = x+self.dropout(new_x)
            x = film_params[:, None, :K] * x + film_params[:, None, K:]
            y= self.norm2(x)
            y =self.feedforward(y)
            return (x+y)
        else:
            x = film_params[:, None, :K] * x + film_params[:, None, K:]
            new_x = self.attention(x,x,x)
            x = x + self.dropout(new_x)
            y = x = self.norm1(x)
            y = self.feedforward(y)
            return self.norm2(x+y)
        
        


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.absorb = config.diffusion.noise_type == "absorb"
        vocab_size = config.diffusion.num_classes + (1 if self.absorb else 0)
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.length)
        self.blocks = nn.ModuleList([
            EncoderLayer(attention=AttentionLayer(d_model=config.model.hidden_size,
                                                  n_heads=config.model.n_heads,
                                                  attention_dropout=config.model.dropout),
                         feedforward=get_ffn_unit(in_dim=config.model.hidden_size,
                                                  ffn_unit = config.model.ffn_unit,
                                                  activation = F.gelu,
                                                  ffn_dropout = config.model.dropout),
                         temb_dim = config.model.length,
                         d_model= config.model.hidden_size,
                         dropout = config.model.dropout,
                         prenorm = config.model.prenorm)
            for _ in range(config.model.n_blocks)
        ])  
        self.norm_layer = nn.LayerNorm(config.model.hidden_size)
        self.linear = nn.Linear(config.model.hidden_size,vocab_size)
        
    def forward(self,indices,sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, c)
        x = self.norm_layer(x)
        x = self.linear(x)
        return x
