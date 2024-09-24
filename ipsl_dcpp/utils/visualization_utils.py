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
    if(save and ffmpeg):
        ani.save(f'{file_name}.gif',writer=writer)
    elif(save):
        ani.save(f'{file_name}.gif',fps=1)

    else:
        return ani



def generate_monthly_dates(start_year, start_month, num_months):
    dates = []
    current_year = start_year
    current_month = start_month

    for _ in range(num_months):
        # Create a date for the current year and month
        date = datetime.datetime(current_year, current_month, 1)
        dates.append(date)

        # Increment the month
        current_month += 1
        # If month exceeds 12, reset to January and increment the year
        if current_month > 12:
            current_month = 1
            current_year += 1

    return dates
import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot_with_fill(means, stds, bmeans,bstds,color='skyblue', alpha=0.3,axes=None):
    """
    Plots means with a filled area representing the standard deviations.
    
    Parameters:
    - means: List or array of mean values.
    - stds: List or array of standard deviation values.
    - color: Color of the filled area. Default is 'skyblue'.
    - alpha: Transparency level of the filled area. Default is 0.3.
    """
    x = generate_monthly_dates(1961,1,len(means))
    import matplotlib.dates as mdates
    # myFmt = mdates.DateFormatter('%M-%Y')
    # fig,axes=plt.subplots(1,figsize=(12, 6))
    # x = np.arange(len(means))  # X values (e.g., 0, 1, 2, ..., len(means)-1)
    # base = datetime.datetime.today()
    # x = [base - datetime.timedelta(month=1) for x in range(590)]
    # Calculate the upper and lower bounds of the shaded area
    pred_upper_bound = means + stds
    pred_lower_bound = means - stds

    pred_upper_bound2 = means + stds*2
    pred_lower_bound2 = means - stds*2
    batch_upper_bound = bmeans + bstds
    batch_lower_bound = bmeans - bstds
    # axes.xaxis.set_major_formatter(myFmt)

    # Plot the means
    axes.plot(x, means.cpu(), color='red', label='Predicted Mean')
    axes.plot(x, bmeans.cpu(), color='blue', label='IPSL Mean')

    # Fill the area between the upper and lower bounds
    axes.fill_between(x, pred_lower_bound.cpu(), pred_upper_bound.cpu(), color='red', alpha=alpha, label='±1 Pred Std Dev')
    # plt.fill_between(x, pred_lower_bound2, pred_upper_bound2, color='pink', alpha=alpha, label='±2 Std Dev')

    axes.fill_between(x, batch_upper_bound.cpu(), batch_lower_bound.cpu(), color='skyblue', alpha=alpha, label='±1 IPSL Std Dev')
    # plt.fill_between(x, batch_upper_bound, batch_lower_bound, color='lightblue', alpha=alpha, label='±1 Std Dev')
    # years = mdates.YearLocator()   # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%d')
    

    dtFmt = mdates.DateFormatter('%Y') # define the formatting

    axes.xaxis.set_major_locator(mdates.MonthLocator(interval=48))
    # axes.xaxis.set_major_formatter(years_fmt)
    # axes.xaxis.set_minor_locator(months)
    # Add labels and legend
    # plt.xlabel('Month')
    # plt.ylabel('Temperature at Surface')
    # plt.title('Mean and Standard Deviation of Temperature At Surface')
    # plt.legend()
    # datemin = np.datetime64('1960', 'Y')
    # datemax = np.datetime64('1960', 'Y') + np.timedelta64(59, 'Y')
    # axes.format_xdata = mdates.DateFormatter('%Y-%m')
    
    axes.xaxis.set_major_formatter(dtFmt) 
    # show every 12th tick on x axes
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=0, fontweight='light')
    # axes.set_xticks(rotation=0, fontweight='light',  fontsize='x-small',ticks=)
    # axes.set_xlim(datemin, datemax)
    # Show the plot
    # plt.show()
    # plt.savefig('long_rollout.png')
    return axes

