from torch import nn 
from attention_block import BasicLayer
from embedding import PatchEmbed2D,PatchEmbed3D
import numpy as np
import torch
from ipsl_dataset import surface_variables,plev_variables,depth_variables
import lightning.pytorch as pl
from patch_recovery import PatchRecovery2D,PatchRecovery3D
from timm.models.layers import trunc_normal_, DropPath
from sampling import UpSample,DownSample




class PanguWeather(nn.Module):
    def __init__(self, 
                 lon_resolution=144,
                 lat_resolution=143,
                 emb_dim=192, 
                 num_heads=(6, 12, 12, 6),
                 patch_size=(2, 4, 4),
                 two_poles=False,
                 window_size=(2, 6, 12), 
                 depth_multiplier=1,
                 position_embs_dim=0,
                 surface_ch=len(surface_variables),
                 level_ch=len(plev_variables),
                 depth_ch=len(depth_variables),
                 use_prev=False, 
                 use_skip=False,
                 conv_head=False,
                 soil=False,
                delta=False):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(0, 0.2, 8*depth_multiplier).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        #self.zdim = 8 if patch_size[0]==2 else 5 # 5 for patch size 4
        self.soil = soil
        self.zdim = 17 if self.soil else 11

        #this is some off by one error where the padding makes up for the fact that this is in fact 35x36 if i use lat_resolution and lon_resolution
        self.layer1_shape = (self.lon_resolution//self.patch_size[1], self.lon_resolution//self.patch_size[2])
        
        self.layer2_shape = (self.layer1_shape[0]//2, self.layer1_shape[1]//2)
        
        self.positional_embeddings = nn.Parameter(torch.zeros((position_embs_dim, lat_resolution, lon_resolution)))
        torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)
        
        self.loss = torch.nn.MSELoss()

        self.patchembed2d = PatchEmbed2D(
            img_size=(lat_resolution, lon_resolution),
            patch_size=patch_size[1:],
            in_chans=surface_ch + position_embs_dim + 1,  # for exta constant later
            embed_dim=emb_dim,
        )
        self.plev_patchembed3d = PatchEmbed3D(
            img_size=(19, lat_resolution, lon_resolution),
            patch_size=patch_size,
            in_chans=level_ch,
            embed_dim=emb_dim
        )
        self.depth_patchembed3d = PatchEmbed3D(
            img_size=(11, lat_resolution, lon_resolution),
            patch_size=patch_size,
            in_chans=depth_ch,
            embed_dim=emb_dim
        )
        self.layer1 = BasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier]
        )
        self.downsample = DownSample(in_dim=emb_dim, input_resolution=(self.zdim, *self.layer1_shape), output_resolution=(self.zdim, *self.layer2_shape))
        self.layer2 = BasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:]
        )
        self.layer3 = BasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:]
        )
        self.upsample = UpSample(emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape))
        out_dim = emb_dim if not self.use_skip else 2*emb_dim
        self.layer4 = BasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier]
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D((lat_resolution, lon_resolution), patch_size[1:], out_dim, surface_ch)
        self.plev_patchrecovery3d = PatchRecovery3D((19, lat_resolution, lon_resolution), patch_size, out_dim, level_ch)
        self.depth_patchrecovery3d = PatchRecovery3D((11, lat_resolution, lon_resolution), patch_size, out_dim, depth_ch)

     #   if conv_head:
     #       self.patchrecovery = PatchRecovery3(input_dim=self.zdim*out_dim, output_dim=69, downfactor=patch_size[-1])

    def forward(self, batch, **kwargs):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        surface = batch['state_surface'].squeeze(-3)
        upper_air = batch['state_level']
        depth = batch['state_depth']
        constants = batch['state_constant'].squeeze(-3)
        #what does none here mean? 
        pos_embs = self.positional_embeddings[None].expand((surface.shape[0], *self.positional_embeddings.shape))
        surface = torch.concat([surface, pos_embs,constants], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.plev_patchembed3d(upper_air)
        depth = self.depth_patchembed3d(depth)
        if(self.soil):
            x = torch.concat([surface.unsqueeze(2), upper_air,depth], dim=2)
        else:
            x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        #print('after concat',x.shape)
        x = x.reshape(B, C, -1).transpose(1, 2)
       # print('right before layer1',x.shape)
        x = self.layer1(x)

        skip = x
        x = self.downsample(x)
      #  print('downsampled',x.shape)
        #it = int'lead_time_hours', 24)//24)
        
        x = self.layer2(x)
        x = self.layer3(x)
      #  print('after layer3',x.shape)
        #it3 = (batch['lead_time_hours'] == 72)[..., None].float().expand_as(x)
        
        #x72 = x
        #for i in range(2):
        #    x72 = self.layer2(x72)
        #    x72 = self.layer3(x72)
            
        #x = x*(1-it3) + x72*it3
        latent = x    
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x)
        output = x
        #what is zdim here? output channels
        output = output.transpose(1, 2).reshape(output.shape[0], -1, self.zdim, *self.layer1_shape)
        if not self.conv_head:
            output_surface = output[:, :, 0, :, :]
            if(self.soil):
                output_upper_air = output[:, :, 1:-6, :, :]
                output_depth = output[:,:,-6:,:,:]
                output_surface = self.patchrecovery2d(output_surface)
                output_level = self.plev_patchrecovery3d(output_upper_air)
                output_depth = self.depth_patchrecovery3d(output_depth)
                out = dict(latent=latent,
                            next_state_level=output_level, 
                            next_state_surface=output_surface,
                            next_state_depth=output_depth)
            else:
                output_upper_air = output[:,:,1:,:,:]
                output_surface = self.patchrecovery2d(output_surface)
                output_level = self.plev_patchrecovery3d(output_upper_air)
                out = dict(latent=latent,
                            next_state_level=output_level, 
                            next_state_surface=output_surface,
                             next_state_depth=torch.empty(0))              
               # print('output_surface_shape',output_surface.shape)
               # print('output_upper_air',output_upper_air.shape)

        else:
            output_level, output_surface = self.patchrecovery(output)
           # print('output_level',output_level.shape)
          #  print('output_surface',output_surface.shape)
            out = dict(latent=latent,
                        next_state_level=output_level, 
                        next_state_surface=output_surface,
                        next_state_depth=output_depth)
        return out
