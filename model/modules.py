import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SingleStageTCN
from .impl.asformer import MyTransformer
from .impl.former import Encoder, Norm
from .utils import masked_softmax


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)


class VanillaEncoderPrediction(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_encoders=3, heads=8, dropout=0.1, use_pe=True):
        super().__init__()
        self._encoder = Encoder(
            hidden_dim, 
            num_encoders, 
            heads, 
            dropout=dropout,
            use_pe=use_pe)
        self._out = nn.Linear(hidden_dim, num_classes)
        self._dropout = nn.Dropout()
        
    def forward(self, x):
        
        # Forward without mask
        # x: batch x frames x dim
        # mask: None

        e_out = self._encoder(x)
        out = self._out(self._dropout(e_out))
        return out
    
class ObjectFusion(nn.Module):
    def __init__(self, 
                 env_dim, obj_dim, 
                 hidden_dim,
                 num_encoders, heads,
                 max_obj, dropout=0.1):
        super().__init__()

        self._env_dim = env_dim
        self._obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        self._max_obj = max_obj

        self._env_linear = nn.Linear(env_dim, hidden_dim)
        self._obj_linear = nn.Linear(obj_dim, hidden_dim)
        self._env_norm = Norm(env_dim)
        self._obj_norm = Norm(obj_dim)
        self._obj_env_norm = Norm(hidden_dim)

        self._obj_fuser = Encoder(
            hidden_dim, 
            num_encoders, 
            heads, 
            use_pe=False,
            dropout=dropout)
        self._env_obj_fuser = Encoder(
            hidden_dim,
            num_encoders,
            heads,
            use_pe=False,
            dropout=dropout)

    def fuse_obj(self, env_feat, obj_feat, obj_mask):
        assert(env_feat.size()[-1] == obj_feat.size()[-1]), \
            'Hidden dimension of environment and object must be the same'
        
        batch_size, frames, max_obj, hidden_dim = obj_feat.size()

        # Broadcast all env feat to obj feat
        obj_env_feat = torch.unsqueeze(env_feat, 2) + obj_feat  
        obj_fused_feat = torch.zeros(batch_size, frames, hidden_dim).cuda()
        if (max_obj == 0):
            return obj_fused_feat
        
        # Step each frame
        for begin in range(0, frames):
            end = begin + 1

            # Single frame: batch x max_obj x dim
            # Single mask: batch x max_obj
            frame_oe_feat = obj_env_feat[:, begin:end].contiguous().view(-1, max_obj, hidden_dim)
            mask = obj_mask[:, begin:end].contiguous().view(-1, max_obj)

            # Hard-attention mask
            l2_norm = torch.norm(frame_oe_feat, dim=-1)     # score: batch x max_obj
            l2_norm_softmax = masked_softmax(l2_norm, mask) # softmax: batch x max_obj

            # Adaptive threshold = 1/nObject
            # Output: batch
            esp = 1e-8
            adaptive_thresh = torch.clamp(1. / (esp + torch.sum(mask, dim=-1, keepdim=True)), 0., 1.)

            # Create mask for hard-attn
            # Output: batch x max_obj
            hard_attn_mask = l2_norm_softmax >= adaptive_thresh

            # Choose batch to keep, just keep the batch has more than 1 object
            # Output: batch
            keep_mask = (torch.sum(hard_attn_mask, dim=-1) > 0)
            keep_idx = torch.masked_select(
                torch.arange(hard_attn_mask.size(0)).cuda(),
                keep_mask
            )

            # Get object feature
            # Output: max_obj x batch x hidden_dim
            fuser_input = obj_feat[:, begin:end].contiguous().view(-1, max_obj, hidden_dim)

            if (len(keep_idx)>0):
                fuser_input = fuser_input[keep_idx]         # batch x max_obj x hidden_dim
                hard_attn_mask = hard_attn_mask[keep_idx]   # batch x max_obj

                # Pass to encoder
                # Output: batch x max_obj x hidden_dim
                fuser_output = self._obj_fuser(fuser_input, key_padding_mask=~hard_attn_mask)

                # Normalize result over objects
                fuser_output = torch.sum(fuser_output, dim=1) / torch.sum(hard_attn_mask, dim=-1, keepdim=True)

                padded_output = torch.zeros(batch_size, hidden_dim).cuda()
                padded_output[keep_idx] = fuser_output
                obj_fused_feat[:, begin:end] = padded_output.view(batch_size, -1, hidden_dim)
        
        return obj_fused_feat

    # Fuse object feature to environment feature
    # env feature size: batch x frames x env_dim
    # obj feature size: batch x frames x max_obj x obj_dim
    # obj mask size: batch x frames x max_obj
    # project feature size: batch x frames x hidden_dim
    def forward(self, env_feat, obj_feat, obj_mask):
        env_feat = self._env_norm(env_feat) 
        env_feat = self._env_linear(env_feat)
        
        cnt_nan = torch.sum(torch.isnan(env_feat)).item()
        assert cnt_nan == 0, 'Env feat contains nan'

        obj_feat = self._obj_norm(obj_feat) 
        obj_feat = self._obj_linear(obj_feat)
        
        cnt_nan = torch.sum(torch.isnan(obj_feat)).item()
        assert cnt_nan == 0, 'Obj feat contains nan'

        # Fuse object
        # Output: batch x frames x hidden_dim

        obj_fused_feat = self.fuse_obj(env_feat, obj_feat, obj_mask)
        
        cnt_nan = torch.sum(torch.isnan(obj_fused_feat)).item()
        assert cnt_nan == 0, 'Obj fused feat contains nan'
        
        # Fuse environment end fused object feature
        stacked_feat = torch.stack([env_feat, obj_fused_feat], dim=2)
        batch_size, frames, hidden_dim = env_feat.size()        # Not stacked feature
        project_feat = torch.zeros(batch_size, frames, hidden_dim).cuda()

        for begin in range(0, frames):
            end = begin+1

            fuser_input = stacked_feat[:, begin:end].contiguous()       # batch x (hidden*2)
            fuser_input = fuser_input.view(-1, 2, hidden_dim)           # batch x 2 x hidden

            fuser_output = self._env_obj_fuser(fuser_input)             # batch x 2 x hidden
            
            cnt_nan = torch.sum(torch.isnan(fuser_output)).item()
            assert cnt_nan == 0, 'Obj env fused feat contains nan'
        
            fuser_output = torch.mean(fuser_output, dim=1)              # batch x hidden

            project_feat[:, begin:end] = fuser_output.view(batch_size, -1, hidden_dim)


        cnt_nan = torch.sum(torch.isnan(project_feat)).item()
        assert cnt_nan == 0, 'Projected feat contains nan'

        return project_feat
