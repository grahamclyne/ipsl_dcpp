import xarray as xr 
import matplotlib.pyplot as plt 
import os 
from matplotlib import animation
import ffmpeg
import torch
from matplotlib.colors import TwoSlopeNorm

#this doesnt work in jupyterhub notebook ? 
def make_gif(
    data, #needs to be shape [batch and/or preds,time_steps,lat,lon]
    rollout_length,
    var_name,
    var_num,
    file_name,
    save=False):

    #get dummy frame 
    scratch = os.environ['SCRATCH']
    shell = xr.open_dataset(f'{scratch}/batch_with_tos/1976_1_tos_included.nc').isel(time=0)['tas']
    
    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes = axes.flatten()
    container = []
    vmin = torch.min(data)
    vmax = torch.max(data)
    print(vmin,vmax)
    # axes[0].set_aspect('equal')
    # axes[1].set_aspect('equal')

    for time_step in range(rollout_length):
        shell.data = data[0][time_step]

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        line = shell.plot.pcolormesh(ax=axes[0],add_colorbar=False,vmin=torch.min(data),vmax=torch.max(data),norm=norm, cmap="RdBu_r")
        # title = axes[0].text(0.5,
        #                       1.05,
        #                       "Month {}".format(time_step), 
        #                       size=plt.rcParams["axes.titlesize"],
        #                       ha="center", 
        #                       transform=axes[0].transAxes
        #                      )
        axes[0].set_title('')
        title = plt.text(0.4, 3.01,"Month {}".format(time_step))

        
        if(len(data) > 1):
            shell.data = data[1][time_step]
            line1 = shell.plot.pcolormesh(ax=axes[1],add_colorbar=False,vmin=torch.min(data),vmax=torch.max(data),norm=norm, cmap="RdBu_r")
            axes[1].set_title('')

            container.append([line,line1,title])
        else:
            container.append([line,title])
    # im = shell.plot.pcolormesh(ax=axes[0],add_colorbar=False)
    # cb = fig.colorbar(im, cax=axes[0])

    # plt.title(var_name)
    # tx = axes[0].set_title('Frame 0')
    colorbar = fig.colorbar(line, ax=axes[1])
    # def animate(i):
    #     arr = container[0][0]
    #     im = arr        
    #     im.set_clim(vmin,vmax)
    #     tx.set_text('Frame {0};.format(i)')

    # ds = xr.open_dataset(self.dataset.files[0])
    # shell = ds.isel(time=0)
    # fig1, axes1 = plt.subplots(1,2, figsize=(16, 6))
    # axes1 = axes1.flatten()
    # container = []
    # for time_step in range(rollout_length):
    #     # print(np.stack(ensembles[0]['state_surface']).shape)
    #     # print(np.stack(ipsl_ensemble[0]['state_surface']).shape)
    #     shell['tas'].data = rollout_ensemble[0]['state_surface'][time_step][0][var_num].cpu()
    #        # line = ax1.pcolormesh(steps[time_step][0,0,0])
    #     line = shell['tas'].plot.pcolormesh(ax=axes1[0],add_colorbar=False)
    #     shell['tas'].data = ipsl_ensemble[0]['state_surface'][time_step][var_num].cpu()
    #     line1 = shell['tas'].plot.pcolormesh(ax=axes1[1],add_colorbar=False)
    #     title = axes1[0].text(0.5,1.05,"Diffusion Step {}".format(time_step), 
    #                     size=plt.rcParams["axes.titlesize"],
    #                     ha="center", transform=axes1[0].transAxes,)
    #     axes1[0].set_title('Predicted')
    #     axes1[1].set_title('IPSL')
    
    #     container.append([line,line1,title])
    # if(var_num < 10):
    #     plt.title(self.dataset.surface_variables[var_num])
    writer = animation.FFMpegWriter(fps=2)
    ani = animation.ArtistAnimation(fig, container)
    # ani.save(f'{out_dir}/diffusion_comparison_{var_names[var_num][0]}_ffmpeg.gif',writer=writer)


    # writer = animation.FFMpegWriter(fps=2)
    # ani = animation.FuncAnimation(fig, animate)

#    ani = animation.ArtistAnimation(fig, container)
    if(save):
        ani.save(f'{file_name}.gif',writer=writer)
    else:
        return ani
