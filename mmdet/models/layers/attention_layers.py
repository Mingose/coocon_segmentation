import torch.nn as nn
from mmdet.registry import MODELS
#自定义注意力机制算法
from .attention.CBAM import CBAMBlock as _CBAMBlock
from .attention.BAM import BAMBlock as _BAMBlock
from .attention.SEAttention import SEAttention as _SEAttention
from .attention.ECAAttention import ECAAttention as _ECAAttention
from .attention.ShuffleAttention import ShuffleAttention as _ShuffleAttention
from .attention.SGE import SpatialGroupEnhance as _SpatialGroupEnhance
from .attention.A2Atttention import DoubleAttention as _DoubleAttention
from .attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention as _SequentialPolarizedSelfAttention
from .attention.CoTAttention import CoTAttention as _CoTAttention
from .attention.TripletAttention import TripletAttention as _TripletAttention
from .attention.CoordAttention import CoordAtt as _CoordAtt
from .attention.ParNetAttention import ParNetAttention as _ParNetAttention
 
 
@MODELS.register_module()
class CBAMBlock(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(CBAMBlock, self).__init__()
        print("======激活注意力机制模块【CBAMBlock】======")
        self.module = _CBAMBlock(channel = in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)
    
    
@MODELS.register_module()
class BAMBlock(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(BAMBlock, self).__init__()
        print("======激活注意力机制模块【BAMBlock】======")
        self.module = _BAMBlock(channel = in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)
 
 
@MODELS.register_module()
class SEAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(SEAttention, self).__init__()
        print("======激活注意力机制模块【SEAttention】======")
        self.module = _SEAttention(channel = in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)   
 
 
@MODELS.register_module()
class ECAAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(ECAAttention, self).__init__()
        print("======激活注意力机制模块【ECAAttention】======")
        self.module = _ECAAttention(**kwargs)
 
    def forward(self, x):
        return self.module(x)  
 
 
@MODELS.register_module()
class ShuffleAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(ShuffleAttention, self).__init__()
        print("======激活注意力机制模块【ShuffleAttention】======")
        self.module = _ShuffleAttention(channel = in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)
 
 
@MODELS.register_module()
class SpatialGroupEnhance(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(SpatialGroupEnhance, self).__init__()
        print("======激活注意力机制模块【SpatialGroupEnhance】======")
        self.module = _SpatialGroupEnhance(**kwargs)
 
    def forward(self, x):
        return self.module(x)   
    
 
@MODELS.register_module()
class DoubleAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(DoubleAttention, self).__init__()
        print("======激活注意力机制模块【DoubleAttention】======")
        self.module = _DoubleAttention(in_channels, 128, 128,True)
 
    def forward(self, x):
        return self.module(x)  
 
 
@MODELS.register_module()
class SequentialPolarizedSelfAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(SequentialPolarizedSelfAttention, self).__init__()
        print("======激活注意力机制模块【Polarized Self-Attention】======")
        self.module = _SequentialPolarizedSelfAttention(channel=in_channels)
 
    def forward(self, x):
        return self.module(x)   
    
    
@MODELS.register_module()
class CoTAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(CoTAttention, self).__init__()
        print("======激活注意力机制模块【CoTAttention】======")
        self.module = _CoTAttention(dim=in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)  
 
    
@MODELS.register_module()
class TripletAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(TripletAttention, self).__init__()
        print("======激活注意力机制模块【TripletAttention】======")
        self.module = _TripletAttention()
 
    def forward(self, x):
        return self.module(x)      
 
 
@MODELS.register_module()
class CoordAtt(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(CoordAtt, self).__init__()
        print("======激活注意力机制模块【CoordAtt】======")
        self.module = _CoordAtt(in_channels, in_channels, **kwargs)
 
    def forward(self, x):
        return self.module(x)    
 
 
@MODELS.register_module()
class ParNetAttention(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(ParNetAttention, self).__init__()
        print("======激活注意力机制模块【ParNetAttention】======")
        self.module = _ParNetAttention(channel=in_channels)
 
    def forward(self, x):
        return self.module(x)  