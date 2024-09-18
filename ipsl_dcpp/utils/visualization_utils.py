import xarray as xr 
import matplotlib.pyplot as plt 
import os 
from matplotlib import animation
import torch
from matplotlib.colors import TwoSlopeNorm
import numpy as np

#EL NINO RANGE PROOF

def el_nino_proof_of_range():
    #needed to get proper indices for elnino 3.4 range
    
    scratch = os.environ['SCRATCH']
    ds = xr.open_dataset(f'{scratch}/batch_with_tos/1960_1_tos_included.nc')
    
    tos_data = ds['tos'].isel(time=0)
    el_nino_34 = tos_data.where(
        (tos_data.lat < 5) & (tos_data.lat > -5) & (tos_data.lon > 190) & (tos_data.lon < 240), drop=True
    )
    lats = [round(float(x),5) for x in list(tos_data.lat.data)]
    lons = [round(float(x),5) for x in list(tos_data.lon.data)]
    
    lats.index(-3.80282),lats.index(3.80282),lons.index(192.5),lons.index(237.5) #region of nino 3.4


def el_nino_34_index(data):
    #see https://foundations.projectpythia.org/core/xarray/enso-xarray.html
    #and https://psl.noaa.gov/enso/dashboard.html

     #[num_batch_examples (diff IC), num_members, rollout_length,var,lat,lon]

    #for now, take first IC and first member
    tos_el_nino_data = data[0,0,:,9,68:74,77:95]
    tos_el_nino_data.std()
    gb = tos_el_nino_data
    tos_nino34_anom = gb - gb.mean()
    index_nino34 = tos_nino34_anom.mean(axis=(-1,-2))
    tos_el_nino_data.reshape(data.shape[2],-1).mean(axis=1)
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    el_nino_index_34 = moving_average(index_nino34.reshape(data.shape[2],-1).mean(axis=1),5)
    normalized_index_nino34_rolling_mean = el_nino_index_34 / tos_el_nino_data.std()
    
    # plt.fill_between(
    #     [x for x in range(96)],
    #     normalized_index_nino34_rolling_mean >= 0.4,
    #     0.4,
    #     color='red',
    #     alpha=0.9,
    # )
    # plt.fill_between(
    #     normalized_index_nino34_rolling_mean,
    #     normalized_index_nino34_rolling_mean.where(
    #         normalized_index_nino34_rolling_mean <= -0.4
    #     ).data,
    #     -0.4,
    #     color='blue',
    #     alpha=0.9,
    # )

    return normalized_index_nino34_rolling_mean


#this doesnt work in jupyterhub notebook ? need to figure out how to run "module load ffmpeg"
def make_gif(
    data, #needs to be shape [batch and/or preds,time_steps,lat,lon]
    rollout_length,
    var_name,
    file_name,
    save=False,
    denormalized=False,
    ffmpeg=False):

    #get dummy frame 
    scratch = os.environ['SCRATCH']
    shell = xr.open_dataset(f'{scratch}/batch_with_tos/1976_1_tos_included.nc').isel(time=0)['tas']
    
    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes = axes.flatten()
    container = []
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    # print(vmin,vmax)
    # print(data.shape)
    # axes[0].set_aspect('equal')
    # axes[1].set_aspect('equal')

    for time_step in range(rollout_length):
        shell.data = data[0][time_step]
        if(denormalized):
            norm = TwoSlopeNorm(vmin=vmin,vcenter = (vmin+vmax)/2,vmax=vmax)
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        line = shell.plot.pcolormesh(ax=axes[0],add_colorbar=False,norm=norm, cmap="RdBu_r")
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
            line1 = shell.plot.pcolormesh(ax=axes[1],add_colorbar=False,norm=norm, cmap="RdBu_r")
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
    if(ffmpeg):
        writer = animation.FFMpegWriter(fps=1)
    ani = animation.ArtistAnimation(fig, container)
    # ani.save(f'{out_dir}/diffusion_comparison_{var_names[var_num][0]}_ffmpeg.gif',writer=writer)


#    ani = animation.ArtistAnimation(fig, container)
    if(save):
        ani.save(f'{file_name}.gif',writer=writer)
    else:
        return ani


