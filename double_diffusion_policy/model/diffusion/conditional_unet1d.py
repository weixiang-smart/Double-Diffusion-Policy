from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


# class CrossAttentionWithMultiheadAttention(nn.Module):
#     def __init__(self, input_dim, num_heads):
#         super(CrossAttentionWithMultiheadAttention, self).__init__()

#         # ????
#         # self.projection_x1 = nn.Linear(input_dim, projection_dim)
#         # self.projection_x2 = nn.Linear(input_dim, projection_dim)

#         # ???????
#         self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)

#     def forward(self, x1, x2):
#         # x1_proj = self.projection_x1(x1)
#         # x2_proj = self.projection_x2(x2)

#         # ?????,?? MultiheadAttention ???????? (seq_len, batch_size, embedding_dim)
#         x1_proj = x1.permute(1, 0, 2)
#         x2_proj = x2.permute(1, 0, 2)

#         # ?? MultiheadAttention ???????
#         output, _ = self.multihead_attn(query=x1_proj, key=x2_proj, value=x2_proj)

#         # ???????????
#         output = output.permute(1, 0, 2)

#         return output


class CrossAttentionWithMultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossAttentionWithMultiheadAttention, self).__init__()

        # ????
        # self.projection_x1 = nn.Linear(input_dim, projection_dim)
        # self.projection_x2 = nn.Linear(input_dim, projection_dim)

        # ???????
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, x1, x2):
        # x1_proj = self.projection_x1(x1)
        # x2_proj = self.projection_x2(x2)

        # ?????,?? MultiheadAttention ???????? (seq_len, batch_size, embedding_dim)
        x1_proj = x1.permute(2, 0, 1)
        x2_proj = x2.permute(2, 0, 1)

        # ?? MultiheadAttention ???????
        output, _ = self.multihead_attn(query=x1_proj, key=x2_proj, value=x2_proj)

        # ???????????
        output = output.permute(1, 2, 0)

        return output


def _get_out_shape(in_shape, layers, attn=False):
	x = torch.randn(*in_shape).unsqueeze(0)
	if attn:
		return layers(x, x, x).squeeze(0).shape
	else:
		return layers(x).squeeze(0).shape

def orthogonal_init(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = query.shape
        assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.conv_query(query).reshape(N, C, H*W)#.permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H*W)#.permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H*W)#.permute(0, 2, 1)
        attention = k.transpose(1, 2)@q / C**0.5
        attention = attention.softmax(dim=1)
        output = v@attention
        output = output.reshape(N, C, H, W)
        return query + output # Add with query and output


class AttentionBlock(nn.Module):
	def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, contextualReasoning=False):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.norm2 = norm_layer(dim)
		self.norm3 = norm_layer(dim)
		self.attn = SelfAttention(dim[0])
		self.context = contextualReasoning
		temp_shape = _get_out_shape(dim, self.attn, attn=True)
		self.out_shape = _get_out_shape(temp_shape, nn.Flatten())
		self.apply(orthogonal_init)

	def forward(self, query, key, value):
		x = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
		if self.context:
			return x
		else:
			x = x.flatten(start_dim=1)
			return x


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D_human_policy(nn.Module):
    def __init__(self, 
        input_dim,  # action dim
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        horizon = 16
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)  # Unet的尺寸列表 [input_dim,256,512,1024]
        start_dim = down_dims[0] # 256

        #-----------step数编码-----------
        dsed = diffusion_step_embed_dim 
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim     # 条件dim=step_dim+外部条件dim(obs_dim*obs_horizon) 

        in_out = list(zip(all_dims[:-1], all_dims[1:])) # [(input_dim, 256), (256, 512), (512, 1024)]

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1] # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):   #0 (2, 256)   1 (256, 512)    2 (512, 1024)
            is_last = ind >= (len(in_out) - 1)  # F F T
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # 0 (512, 1024)   1 (256, 512)  
            is_last = ind >= (len(in_out) - 1)  # F F
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder # no
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time encoding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])   # 1

        global_feature = self.diffusion_step_encoder(timesteps)  # step encode 1,256


        
        if global_cond is not None:      # concate step with obs
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # ----------没运行--------------------------------
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        # ----------没运行--------------------------------
            
        
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            # print(x.shape, global_feature.shape)
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        x = self.mid_modules[0](x, global_feature)
        x = self.mid_modules[1](x, global_feature)
        middle_feature = x.clone()

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x, middle_feature
    

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,  # action dim
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        horizon = 16
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)  # Unet的尺寸列表 [input_dim,256,512,1024]
        start_dim = down_dims[0] # 256

        #-----------step数编码-----------
        dsed = diffusion_step_embed_dim 
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        #-------------human action encode ----
        dact = 3 * horizon
        diffusion_action_encoder = nn.Sequential(
            nn.Linear(dact, 512),
            nn.Mish(),
            nn.Linear(512, dact),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim     # 条件dim=step_dim+外部条件dim(obs_dim*obs_horizon) 
        
        # duel condition
        cond_dim = cond_dim + dact

        in_out = list(zip(all_dims[:-1], all_dims[1:])) # [(input_dim, 256), (256, 512), (512, 1024)]

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1] # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):   #0 (2, 256)   1 (256, 512)    2 (512, 1024)
            is_last = ind >= (len(in_out) - 1)  # F F T
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # 0 (512, 1024)   1 (256, 512)  
            is_last = ind >= (len(in_out) - 1)  # F F
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        mid_input_dim = 2048 
        num_heads = 2
        cross_attn_multihead = CrossAttentionWithMultiheadAttention(mid_input_dim,  num_heads)
        self.cross_attention = cross_attn_multihead 

        self.diffusion_step_encoder = diffusion_step_encoder
        self.diffusion_action_encoder = diffusion_action_encoder
        self.local_cond_encoder = local_cond_encoder # no
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], huamn_middle_feature = None,
            diff_action = None,
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time encoding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])   # 1

        global_feature = self.diffusion_step_encoder(timesteps)  # step encode 1,256
        diff_action_feature = self.diffusion_action_encoder(diff_action)
        
        if global_cond is not None:      # concate step with obs
            global_feature = torch.cat([
                global_feature, global_cond, diff_action_feature
            ], axis=-1)
        
        # ----------没运行--------------------------------
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        # ----------没运行--------------------------------
            
        
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            # print(x.shape, global_feature.shape)
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        x = self.mid_modules[0](x, global_feature)
        x = self.mid_modules[1](x, global_feature)
        # middle_feature = x.clone()

        x = self.cross_attention(x, huamn_middle_feature)




        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
    





    

