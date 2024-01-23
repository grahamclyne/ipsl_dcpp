from torch import nn 
from attention_block import BasicLayer
from embedding import PatchEmbed2D,PatchEmbed3D
import numpy as np
import torch
from ipsl_dataset import surface_variables,plev_variables,depth_variables
import lightning.pytorch as pl

from timm.models.layers import trunc_normal_, DropPath






class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class PatchRecovery3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]

class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: 2 * in_lat - pad_bottom, pad_left: 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class PanguWeather(nn.Module):
    def __init__(self, 
                 lon_resolution=144,
                 lat_resolution=143,
                 emb_dim=192, 
                 #num_heads=(6, 12, 12, 6),
                 num_heads=(1,1,1,1),
                 patch_size=(2, 4, 4),
                 two_poles=False,
                 window_size=(2, 6, 12), 
                 depth_multiplier=1,
                 position_embs_dim=0,
                 surface_ch=len(surface_variables),
                 level_ch=len(plev_variables),
                 use_prev=False, 
                 use_skip=False,
                 conv_head=False,
                delta=False):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(0, 0.2, 8*depth_multiplier).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        #self.zdim = 8 if patch_size[0]==2 else 5 # 5 for patch size 4
        self.zdim = 11
        
        #this is some off by one error where the padding makes up for the fact that this is in fact 35x36 if i use lat_resolution and lon_resolution
        self.layer1_shape = (self.lon_resolution//self.patch_size[1], self.lon_resolution//self.patch_size[2])
        
        self.layer2_shape = (self.layer1_shape[0]//2, self.layer1_shape[1]//2)
        
        self.positional_embeddings = nn.Parameter(torch.zeros((position_embs_dim, lat_resolution, lon_resolution)))
        torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)
        
        self.loss = torch.nn.MSELoss()

        self.patchembed2d = PatchEmbed2D(
            img_size=(lat_resolution, lon_resolution),
            patch_size=patch_size[1:],
            in_chans=surface_ch + position_embs_dim,  # add
            embed_dim=emb_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(19, lat_resolution, lon_resolution),
            patch_size=patch_size,
            in_chans=level_ch,
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
        self.patchrecovery2d = PatchRecovery2D((lat_resolution, lon_resolution), patch_size[1:], out_dim, 4)
        self.patchrecovery3d = PatchRecovery3D((19, lat_resolution, lon_resolution), patch_size, out_dim, level_ch)
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
        #depth = batch['depth_inputs'].squeeze(-3)

        pos_embs = self.positional_embeddings[None].expand((surface.shape[0], *self.positional_embeddings.shape))
        
        surface = torch.concat([surface, pos_embs], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)
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
        #what is 11 here? output channels
        output = output.transpose(1, 2).reshape(output.shape[0], -1, 11, *self.layer1_shape)
        #print('output shape',output.shape)
        if not self.conv_head:
            output_surface = output[:, :, 0, :, :]
            output_upper_air = output[:, :, 1:, :, :]
            #print('output_surface_shape',output_surface.shape)
           # print('output_upper_air',output_upper_air.shape)
            output_surface = self.patchrecovery2d(output_surface)
            output_level = self.patchrecovery3d(output_upper_air)
            
            output_surface = output_surface.unsqueeze(-3)
            
        else:
            output_level, output_surface = self.patchrecovery(output)
       # print('output_level',output_level.shape)
       # print('output_surface',output_surface.shape)
        out = dict(latent=latent,
                   next_state_level=output_level, 
                    next_state_surface=output_surface)
        return out
