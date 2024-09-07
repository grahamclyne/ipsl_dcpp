from torch import nn 
import torch
from torch.nn import GELU as GeLU

def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)
 



class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x





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
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, (1,2), patch_size)
        # print(img_size, patch_size, in_chans, out_chans)

    def forward(self, x):
        print(x.shape)

        output = self.conv(x)
       # print('output',output.shape)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]
      #  print(h_pad,w_pad)
        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)
       # print(output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right].shape)
        # return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]
        return output[:,:,:,:]

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
    
    
class PatchRecovery3(nn.Module):
    def __init__(self, 
        input_dim=None,
        dim=192,
        downfactor=4,
        output_dim=182,
        soil=True,
        plev=True,
        cropped=False
        ):
# input dim equals input_dim*z since we will be flattening stuff ?
        super().__init__()
        self.downfactor = downfactor
        if input_dim is None:
            input_dim = 8*dim
        self.head1 = nn.Sequential(
            nn.Conv2d(input_dim, 8*dim, kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(8*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            GeLU(),
            nn.Conv2d(4*dim, 4*dim, kernel_size=1, stride=1, padding=0))
          #  nn.Conv2d(4*dim, output_dim, kernel_size=(2,3), stride=1, padding=(0,1)))
       # self.head2 = nn.Sequential(
       #     nn.GroupNorm(num_groups=32, num_channels=4*dim, eps=1e-6, affine=True),
       #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
       #     nn.Conv2d(4*dim, 4*dim, kernel_size=3, stride=1, padding=1),
       #     GeLU())
        self.cropped = cropped
        if(not self.cropped):
            self.norm = nn.LayerNorm((output_dim,143,144))
        else:
            self.norm = nn.LayerNorm((output_dim,144,144))

        if(self.cropped):
            self.proj = nn.Conv2d(4*dim, output_dim, kernel_size=1, stride=1, padding=0)
            
        else:
            self.proj = nn.Conv2d(4*dim, output_dim, kernel_size=(4,5), stride=1, padding=(1,2))
            
        self.soil = soil
        self.plev = plev
        self.output_dim = output_dim

    def forward(self, x):
        #print('before',x.shape)
        #x = x.permute(0, 2, 1)
        #x = x.reshape((x.shape[0], x.shape[1]*Z, H, W))
     #   print(x.shape)
        x = x.flatten(1, 2)
        x = self.head1(x)
       # if self.downfactor == 4:
       #     x = self.head2(x)
     #   print('after head',x.shape)
        x = self.proj(x)
        x = self.norm(x)
     #   print(x.shape)
        #output_surface = x[:, :135]
      #  output = x[:, 135:287].reshape((x.shape[0], 8, 19, *x.shape[-2:]))
      #  output_depth = x[:,287:].reshape((x.shape[0],3,11,*x.shape[-2:]))
      #  print('after',x.shape)
        if(self.cropped):
            output_surface = x[:,:self.output_dim,:143,:]
        else:
            output_surface = x[:,:self.output_dim]
        if(self.soil):
            output_depth = x[:,91:94].reshape((x.shape[0],3,11,*x.shape[-2:]))
        else:
            output_depth = torch.empty(0)
        if(self.plev):
            output_plev = x[:,9:].reshape((x.shape[0],8,19,*x.shape[-2:]))
        else:
            output_plev = torch.empty(0)
        return output_surface,output_depth,output_plev


class PatchRecovery4(nn.Module):
    # thank you Nidhin and https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
    
    def __init__(self, 
        input_dim=192,
        output_dim=182,
                 smoothing=False
        ):
        super().__init__()
        upscale_factor = 2
        self.smoothing=smoothing
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.conv = nn.Conv2d(input_dim, output_dim * (upscale_factor ** 2), 3, 1, 1, bias=0)
        weight = ICNR(self.conv.weight, initializer=nn.init.kaiming_normal_,
                      upscale_factor=upscale_factor)
        self.smooth_conv = nn.Conv2d(output_dim, output_dim, kernel_size=5, stride=1, padding=2)
        self.smooth_conv1 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm((output_dim,144,144))
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x):
        #x == (batch_size, 572, 72,72)
        output = self.conv(x)  
        output = self.pixelshuffle(output)  # (batch_size, 34, 144, 144)
        if(self.smoothing):
            output = self.smooth_conv(output)
            output = self.norm(output)
            output = self.smooth_conv1(output)
            output = self.norm(output)
        output_surface = output[:,:,:143,:] # crop for irregular
        
        return output_surface
