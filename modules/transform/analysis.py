import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, AttentionBlock
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *

# injector, extractor
class Injector(nn.Module) : 
    def __init__(self, dim, text_feature_dim, num_attn_head=8) :
        super().__init__()

        self.text_feature_dim_proj = nn.Linear(text_feature_dim, dim)
        nn.init.kaiming_normal_(self.text_feature_dim_proj.weight)

        self.image_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(dim, num_attn_head, batch_first=True)
        self.gamma = nn.Parameter(0.0 * torch.ones((dim)), requires_grad=True) 
    
    def forward(self, image_features:torch.Tensor, text_features:torch.Tensor) :    
        
        b,c,h,w = image_features.size()
        image_features = image_features.contiguous().flatten(2).permute(0,2,1)

        text_features = self.text_feature_dim_proj(text_features)

        text_features = self.text_norm(text_features)
        image_features = self.image_norm(image_features)                                                                                                                                                                          

        attn_out, attn_weights = self.cross_attn(image_features, text_features, text_features)
        
        image_features = image_features + self.gamma * attn_out
        image_features = image_features.contiguous().permute(0,2,1).view(b,c,h,w)

        return image_features

class Extractor(nn.Module) :
    def __init__(self, dim, text_feature_dim, num_attn_head=8) :
        super().__init__()

        self.image_feature_dim_proj = nn.Linear(dim, text_feature_dim)
        nn.init.kaiming_normal_(self.image_feature_dim_proj.weight)

        self.image_norm = nn.LayerNorm(text_feature_dim)
        self.text_norm = nn.LayerNorm(text_feature_dim)
        self.cross_attn = nn.MultiheadAttention(text_feature_dim, num_attn_head, batch_first=True)
    
    def forward(self, text_features:torch.Tensor, image_features:torch.Tensor) :

        image_features = image_features.contiguous().flatten(2).permute(0,2,1)

        image_features = self.image_feature_dim_proj(image_features)

        text_features = self.text_norm(text_features)
        image_features = self.image_norm(image_features)

        attn_out, attn_weights = self.cross_attn(text_features, image_features, image_features)
        
        text_features = text_features + attn_out

        return text_features

class AnalysisTransformEX(nn.Module):
    def __init__(self, N, M, text_embedding_dim, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            conv(3, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            Injector(N, text_embedding_dim),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            Extractor(N, text_embedding_dim),
            Injector(N, text_embedding_dim),
            conv(N, M),
            AttentionBlock(M)
        )

    def forward(self, x, text_embeddings):
        
        for layer in self.analysis_transform:
            if type(layer) == Injector :
                x = layer(x, text_embeddings)
            elif type(layer) == Extractor :
                text_embeddings = layer(text_embeddings, x)
            else:
                x = layer(x)
                
        return x

class HyperAnalysisEX(nn.Module):
    def __init__(self, N, M, act=nn.ReLU) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            act(),
            conv(N, N),
            act(),
            conv(N, N)
        )

    def forward(self, x):
        x = self.reduction(x)
        return x

