from torch import nn 
from ipsl_dcpp.model.attention_block import BasicLayer,CondBasicLayer
from ipsl_dcpp.model.embedding import PatchEmbed2D,PatchEmbed3D
import numpy as np
import torch
#from ipsl_dataset import surface_variables,plev_variables,depth_variables
import lightning.pytorch as pl
from ipsl_dcpp.model.patch_recovery import PatchRecovery2D,PatchRecovery3D,PatchRecovery3
from timm.models.layers import trunc_normal_, DropPath
from ipsl_dcpp.model.sampling import UpSample,DownSample
import datetime
import math
#from pudb import set_trace; set_trace()



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
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





class PanguWeather(nn.Module):
    def __init__(self, 
                 lon_resolution,
                 lat_resolution,
                 emb_dim, 
                 cond_dim, 
                 num_heads,
                 patch_size,
                 two_poles,
                 window_size, 
                 depth_multiplier,
                 position_embs_dim,
                 surface_ch,
                 level_ch,
                 depth_ch,
                 use_skip,
                 conv_head,
                 soil,
                delta,
                plev
                ):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(0, 0.2, 8*depth_multiplier).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        #self.zdim = 8 if patch_size[0]==2 else 5 # 5 for patch size 4
        self.soil = soil
        self.zdim = 1 
      #  self.zdim = zdim + 6 if self.soil else 11
        
    #    self.zdim = 7 if self.soil else 1
        #this is some off by one error where the padding makes up for the fact that this is in fact 35x36 if i use lat_resolution and lon_resolution
        self.layer1_shape = (self.lon_resolution//self.patch_size[1], self.lon_resolution//self.patch_size[2])
        
        self.layer2_shape = (self.layer1_shape[0]//2, self.layer1_shape[1]//2)
        
        #self.positional_embeddings = nn.Parameter(torch.zeros((position_embs_dim, lat_resolution, lon_resolution)))
        #torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)
      #  self.time_embedding = TimestepEmbedder(cond_dim)
        #nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        #nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)
        self.loss = torch.nn.MSELoss()

        self.patchembed2d = PatchEmbed2D(
            img_size=(lat_resolution, lon_resolution),
            patch_size=patch_size[1:],
            in_chans=surface_ch,  
            embed_dim=emb_dim,
        )
        # self.plev_patchembed3d = PatchEmbed3D(
        #     img_size=(19, lat_resolution, lon_resolution),
        #     patch_size=patch_size,
        #     in_chans=level_ch,
        #     embed_dim=emb_dim
        # )
        # self.depth_patchembed3d = PatchEmbed3D(
        #     img_size=(11, lat_resolution, lon_resolution),
        #     patch_size=patch_size,
        #     in_chans=depth_ch,
        #     embed_dim=emb_dim
        # )
        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier]
        )
        self.downsample = DownSample(in_dim=emb_dim, input_resolution=(self.zdim, *self.layer1_shape), output_resolution=(self.zdim, *self.layer2_shape))
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:]
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            cond_dim = cond_dim,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:]
        )
        self.upsample = UpSample(emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape))
        out_dim = emb_dim if not self.use_skip else 2*emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier]
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.

        self.emb_dim = emb_dim
        output_dim = int((surface_ch * 1)/2)

        if conv_head:
            if(self.soil):
                output_dim += (depth_ch * 11)
            if(self.plev):
                output_dim += (level_ch * 19)
            #print(output_dim)
            self.patchrecovery = PatchRecovery3(input_dim=self.zdim*out_dim, output_dim=output_dim,soil=self.soil,plev=self.plev)

        else:
            self.patchrecovery2d = PatchRecovery2D((lat_resolution, lon_resolution), patch_size[1:], out_dim, output_dim)
          #  self.plev_patchrecovery3d = PatchRecovery3D((19, lat_resolution, lon_resolution), patch_size, out_dim, level_ch)
            self.depth_patchrecovery3d = PatchRecovery3D((11, lat_resolution, lon_resolution), patch_size, out_dim, depth_ch)
    def forward(self, batch,c, **kwargs):
        """
        Args:
            surface (torch.Tensor): 2D 
            surface_mask (torch.Tensor): 2D 
            upper_air (torch.Tensor): 3D 
        """
        surface = batch['state_surface'].squeeze(-4)
        if(self.plev):
            upper_air = batch['state_level'].squeeze(-5)
        if(self.soil):
            depth = batch['state_depth'].squeeze(-5)
        constants = batch['state_constant'].squeeze(-4)
        #forcings = batch['forcings']
       # dt = np.vectorize(datetime.datetime.strptime)(batch['time'],'%Y-%m')
       # time_step_conversion = np.vectorize(lambda x: x.month)
       # timestep = torch.Tensor(time_step_conversion(dt)).to(surface.device)
        
       # c = self.time_embedding(timestep)
        #pos_embs = self.positional_embeddings[None].expand((surface.shape[0], *self.positional_embeddings.shape))
        surface = self.patchembed2d(surface)
        x = surface.unsqueeze(2)
        #print(x.shape,'after unsqueeze,and embedding')
        if(self.soil):
            depth = self.depth_patchembed3d(depth)
            x = torch.concat([x,depth], dim=2)
        if(self.plev):
            upper_air = self.plev_patchembed3d(upper_air)
            x = torch.concat([x,upper_air], dim=2)
        
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = self.layer1(x,c)
        skip = x
        x = self.downsample(x)
        
        x = self.layer2(x,c)
        x = self.layer3(x,c)
        latent = x    
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x,c)
        output = x
        #what is zdim here? output channels
        
        output = output.transpose(1, 2).reshape(output.shape[0], -1, self.zdim, *self.layer1_shape)
        if not self.conv_head:
            output_surface = output[:, :, 0, :, :]
            if(self.soil):
                #output_upper_air = output[:, :, 1:-6, :, :]
               # output_depth = output[:,:,-6:,:,:]
                output_depth = output[:,:,1:,:,:]
                output_surface = self.patchrecovery2d(output_surface)
               # output_level = self.plev_patchrecovery3d(output_upper_air)
                output_depth = self.depth_patchrecovery3d(output_depth)
                out = dict(latent=latent,
                #            next_state_level=output_level, 
                            next_state_surface=output_surface,
                            next_state_depth=output_depth)
            else:
          #      output_upper_air = output[:,:,1:,:,:]
                output_surface = self.patchrecovery2d(output_surface)
          #      output_level = self.plev_patchrecovery3d(output_upper_air)
                out = dict(latent=latent,
                            next_state_level=torch.empty(0), 
                            next_state_surface=output_surface,
                             next_state_depth=torch.empty(0))              
               # print('output_surface_shape',output_surface.shape)
               # print('output_upper_air',output_upper_air.shape)

        else:
         #   output_level, output_surface,output_depth = self.patchrecovery(output)
            output_surface,output_depth,output_level = self.patchrecovery(output)
            out = dict(latent=latent,
                        next_state_level=output_level, 
                       state_surface=batch['state_surface'],
                       state_depth=batch['state_depth'],
                        next_state_surface=output_surface,
                        next_state_depth=output_depth)
        return out
