import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as tick
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy
import cartopy.crs as ccrs
cartopy.config['data_dir'] = 'apps/cartopy-data'
import cartopy.feature as cfeature

from model_predictions import predict_ensemble, deterministic_predict
from inference_metrics import fft_mean, calculate_normalized_rank_histogram_xs, plot_normalized_rank_histogram_xs, calculate_roc_for_ensemble, plot_roc_curves 
#------------------------------------------------------------- 
# load input and target data
#-------------------------------------------------------------
era5 = xr.open_zarr('ERA5_ML_data.zarr')
# Load the test period ERA5 data over tasmania domain
e5 = era5['ERA5'].sel(latitude=slice(-36.75, -46), longitude=slice(142, 151.25)).sel(time=slice('2021-01-01T00:00:00.000000000', 
                              '2022-12-31T23:00:00.000000000'))
# Load the static data
x_static = xr.open_zarr('B_C2_static.zarr')['BC2_static']
# Load the test period BARRA-C2 precipitation data over tasmania domain
b2c = xr.open_zarr('B2C_Pr_data_2k25.zarr')
b2c_pr = b2c['pr'].sel(time=slice('2021-01-01T00:00:00.000000000', 
                              '2022-12-31T23:30:00.000000000')).expand_dims(channel=1, axis=-1)

qs = xr.open_zarr('ERA5_pr_data_tas.zarr')
era5 = qs.tp.sel(latitude=slice(-39.8, -44.25), longitude=slice(144.3, 148.75)).sel(time=slice('2021-01-01T00:00:00.000000000', 
                              '2022-12-31T23:00:00.000000000'))
n_steps=17520
era5_input = era5[0:n_steps,...]
#------------------------------------------------------------- 
# Run the ensemble prediction function
#------------------------------------------------------------- 
ens_pred, target = predict_ensemble(n_steps=17520, mask=None)
#------------------------------------------------------------- 
# Create a figure with two subplots: one for the histogram and one for the PSD plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

linewidth = 1
bins = 100
font_size = 16  # Set font size for labels, titles, and ticks
legend_font_size = 12  # Font size for the legend

# Subplot (a): Histogram plot
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

labels = ['R3GAN-115[1.1, 0]', 'R3GAN-115[1.1, 1]', 'R3GAN-121[1.0, 0]', 'R3GAN-121[1.0, 1]', 
          'R3GAN-123[1.0, 0]', 'R3GAN-123[1.0, 1]', 'R3GAN-125[1.0, 0]', 'R3GAN-125[1.0, 1]', 
          'R3GAN-129[1.15, 0]', 'R3GAN-129[1.15, 1]']
'''
labels = ['R3GAN-129[0.75, 9]', 'R3GAN-129[0.80, 9]', 'R3GAN-129[0.85, 9]', 'R3GAN-129[0.90, 9]', 
          'R3GAN-129[0.95, 9]', 'R3GAN-129[1.00, 9]', 'R3GAN-129[1.05, 9]', 'R3GAN-129[1.10, 9]', 
          'R3GAN-129[1.15, 9]', 'R3GAN-129[1.20, 9]']

labels = ['R3GAN-129[1.15, 0]', 'R3GAN-129[1.15, 1]', 'R3GAN-129[1.15, 2]', 'R3GAN-129[1.15, 3]', 
          'R3GAN-129[1.15, 4]', 'R3GAN-129[1.15, 5]', 'R3GAN-129[1.15, 6]', 'R3GAN-129[1.15, 7]', 
          'R3GAN-129[1.15, 8]', 'R3GAN-129[1.15, 9]']
'''

n_ensemble = 10
for i in range(n_ensemble):
    ax1.hist(ens_pred[...,i].values.flatten(), bins=bins, alpha=0.5, histtype='step', linewidth=linewidth, color=colors[i], label=labels[i])


# Plot the histogram for Target with a thick line and a different color
ax1.hist(target.values.flatten(), bins=bins, alpha=0.5, label='BARRA-C2', histtype='step', linewidth=linewidth+1, color='black')

# Plot the histogram for ERA5 with a thick line and another different color
ax1.hist(era5_input.values.flatten(), bins=bins, alpha=0.5, label='ERA5', histtype='step', linewidth=linewidth+1, color='saddlebrown')

# Set log scale for y-axis
ax1.set_yscale('log')

# Add labels, legend, and title for the histogram
ax1.set_xlabel('Precipitation (mm/hr)', fontsize=font_size)
ax1.set_ylabel('Counts', fontsize=font_size)
ax1.legend(fontsize=legend_font_size)
ax1.set_title('(e) Distribution', fontsize=font_size+2)

# Increase tick font size
ax1.tick_params(axis='both', labelsize=font_size-1)

ax1.grid(True)

# Subplot (b): PSD plot
# Calculate the x-values for the wavelength
xval = 112 * 4.4 / 2 / np.arange(1, 57)

# Plot each GAN member with thin and dashed lines of the same color, but without individual legend labels
for i in range(n_ensemble):
    ax2.plot(xval, np.diag(np.log10(fft_mean(ens_pred[...,i].compute()))[56:, 56:]),  linewidth=linewidth, color=colors[i], label=labels[i])


# Plot the target data with a thick line
ax2.plot(xval, np.diag(np.log10(fft_mean(target.compute()))[56:, 56:]), label='BARRA-C2', linewidth=linewidth+1, color='black')

# Invert x-axis and apply log scale for x-axis
ax2.invert_xaxis()
ax2.set_xscale('log', base=2)

# Set labels and ticks for the PSD plot
ax2.set_xlabel(r'Wavelength (λ/2) [km]', fontsize=font_size)
ax2.set_ylabel('Log10(PSD)', fontsize=font_size)

# Custom x-ticks
xticks = [112 * 4.4 / 2 / i for i in [1, 2, 4, 8, 16, 32, 56]]  # Example x-tick locations
ax2.set_xticks(xticks)
ax2.set_xticklabels([f'{xt:.1f}' for xt in xticks], fontsize=font_size)

# Add a legend and title for the PSD plot
ax2.legend(fontsize=legend_font_size)
ax2.set_title('(f) Power Spectral Density (PSD)', fontsize=font_size+2)

# Increase tick font size
ax2.tick_params(axis='both', labelsize=font_size-1)

ax2.grid(True)

# Add main title
fig.suptitle("Ensemble-3", fontsize=font_size+3)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Dist_PSD_R3GAN8_ens_3.pdf', bbox_inches="tight")
#-------------------------------------------------------------
# Normalised rank histogram plot
#------------------------------------------------------------- 
# plot normalised rank histogram
norm_rank_hist = calculate_normalized_rank_histogram_xs(ens_pred, target)
plot_normalized_rank_histogram_xs(norm_rank_hist)
#------------------------------------------------------------- 
# ROC curve
#------------------------------------------------------------- 
# Define thresholds
thresholds = [0.25, 1.0]
# Plot ROC curves
plot_roc_curves(thresholds, ens_pred, target)
#------------------------------------------------------------- 
# Qualitative analysis - ensemble spatial plot for an event
#------------------------------------------------------------- 
# Define the time range
hours = ['2021-05-25T05', '2021-05-25T06', 
         '2021-05-25T07', '2021-05-25T08', 
         '2021-05-25T09', '2021-05-25T10']       
'''
hours = ['2022-10-13T07', '2022-10-13T08', 
         '2022-10-13T09', '2022-10-13T10', 
         '2022-10-13T11', '2022-10-13T12']
'''
# Define colors to match your colorbar image
colors = [
    (1, 1, 1),           # White (0.01 - 0.5)
    (0.17, 0.68, 0.82),  # Cyan
    (0.09, 0.49, 0.76),  # Light blue
    (0, 0.27, 0.70),     # Blue
    (0.12, 0.56, 0.22),  # Light green
    (0.17, 0.70, 0.29),  # Green
    (0.99, 0.92, 0.25),  # Yellow
    (0.97, 0.62, 0.20),  # Orange
    (0.89, 0.07, 0.19),  # Red
    (0.54, 0, 0),        # Dark red
    (0.54, 0.27, 0.07)   # Brown
]

# Compute vmax dynamically from target data
vmax = max([target.sel(time=hour).compute().max().values for hour in hours])

# Update clevels to include 0.5 as a break point
clevels = np.concatenate([np.linspace(-0.0001, 0.5, 1), np.linspace(0.5, vmax, 10)])
# Create the colormap and normalization
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(clevels)-1)
norm = mpl.colors.BoundaryNorm(clevels, cmap.N, clip=True)

# Create the figure with subplots
fig, axs = plt.subplots(len(hours), 12, figsize=(28, 3 * len(hours)), 
                         subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

labels = ['R3GAN-115[1.1, 0]', 'R3GAN-115[1.1, 1]', 'R3GAN-121[1.0, 0]', 'R3GAN-121[1.0, 1]', 
          'R3GAN-123[1.0, 0]', 'R3GAN-123[1.0, 1]', 'R3GAN-125[1.0, 0]', 'R3GAN-125[1.0, 1]', 
          'R3GAN-129[1.15, 0]', 'R3GAN-129[1.15, 1]']
'''
labels = ['R3GAN-129[0.75, 9]', 'R3GAN-129[0.80, 9]', 'R3GAN-129[0.85, 9]', 'R3GAN-129[0.90, 9]', 
          'R3GAN-129[0.95, 9]', 'R3GAN-129[1.00, 9]', 'R3GAN-129[1.05, 9]', 'R3GAN-129[1.10, 9]', 
          'R3GAN-129[1.15, 9]', 'R3GAN-129[1.20, 9]']
labels = ['R3GAN-129[1.15, 0]', 'R3GAN-129[1.15, 1]', 'R3GAN-129[1.15, 2]', 'R3GAN-129[1.15, 3]', 
          'R3GAN-129[1.15, 4]', 'R3GAN-129[1.15, 5]', 'R3GAN-129[1.15, 6]', 'R3GAN-129[1.15, 7]', 
          'R3GAN-129[1.15, 8]', 'R3GAN-129[1.15, 9]']
'''
fontsize = 16
for i, hour in enumerate(hours):
    era5_data = era5_input.sel(time=hour)
    target_data = target.sel(time=hour)
    
    # Compute all 11 ensemble members using dask
    ensemble_members = [ens_pred[..., j].sel(time=hour) for j in range(10)]
    computed_data = dask.compute(era5_data, *ensemble_members, target_data)

    # Plot ERA5
    plot = computed_data[0].plot(ax=axs[i, 0], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 0].set_title(f"ERA5 \n{hour}", fontsize=fontsize)
    else:
        axs[i, 0].set_title(f"{hour}", fontsize=fontsize)

    # Plot all ensemble members
    for j in range(10):
        computed_data[j + 1].plot(ax=axs[i, j + 1], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
        if i == 0:
            axs[i, j + 1].set_title(labels[j], fontsize=fontsize)
        else:
            axs[i, j + 1].set_title("")

    # Plot BARRA-C2
    plot1 = computed_data[-1].plot(ax=axs[i, 11], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 11].set_title("BARRA-C2", fontsize=fontsize)
    else:
        axs[i, 11].set_title("")

    # Add coastlines
    for ax in axs[i]:
        ax.coastlines(resolution='10m', color='black', linewidth=1)

# Add a single colorbar with the correct limits
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=axs, location='bottom', fraction=0.03, pad=0.01, shrink=1.25, format=tick.FormatStrFormatter('%.2f'))
cbar.set_label('Precipitation (mm/hr)', fontsize=20)
# Increase the size of the colorbar tick labels
cbar.ax.tick_params(labelsize=18)

# Add main title
fig.suptitle("Ensemble-3", fontsize=20)

plt.savefig('SP_EVENT-1_R3GAN8_ens_3.pdf')
#------------------------------------------------------------- 
# Compute CRPS and Brier scores using xskillscore
#-------------------------------------------------------------
crps = xs.crps_ensemble(target, ens_pred, member_dim="member")
print('CRPS: ', crps.values)
threshold = [0.25, 1.0, 5.0]
brier_scores = xs.threshold_brier_score(target, ens_pred, threshold, member_dim="member")
print(f'Brier_score-{threshold[0]}: ', brier_scores[0].values, f'Brier_score-{threshold[1]}: ', brier_scores[1].values,
     f'Brier_score-{threshold[2]}: ', brier_scores[2].values)
#------------------------------------------------------------- 
Unet_pr, GAN_pr, BARRAC2_pr = deterministic_predict(n_steps=17520, mask=None)
#-------------------------------------------------------------
# Deterministic distribution and PSD plot
#-------------------------------------------------------------

# Create a figure with two subplots: one for the histogram and one for the PSD plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

linewidth = 1
bins = 100
font_size = 16
# Subplot (a): Histogram plot

# Plot the histogram for Target with a thick line and a different color
ax1.hist(BARRAC2_pr.values.flatten(), bins=bins, alpha=0.5, label='BARRA-C2', histtype='step', linewidth=linewidth+1, color='black')

# Plot the histogram for ERA5 with a thick line and another different color
ax1.hist(era5_input.values.flatten(), bins=bins, alpha=0.5, label='ERA5', histtype='step', linewidth=linewidth+1, color='saddlebrown')

# Plot the histogram for Unet with a thick line and another different color
ax1.hist(Unet_pr.values.flatten(), bins=bins, alpha=0.5, label='UNET', histtype='step', linewidth=linewidth+1, color='tab:blue')

# Plot the histogram for GAN with a thick line and another different color
ax1.hist(GAN_pr.values.flatten(), bins=bins, alpha=0.5, label='R3GAN', histtype='step', linewidth=linewidth+1, color='tab:orange')

# Set log scale for y-axis
ax1.set_yscale('log')

# Add labels, legend, and title for the histogram
ax1.set_xlabel('Precipitation (mm/hr)', fontsize=font_size)
ax1.set_ylabel('Counts', fontsize=font_size)
ax1.legend(fontsize=font_size)
ax1.set_title('(a) Distribution', fontsize=font_size+2)

# Increase tick font size
ax1.tick_params(axis='both', labelsize=font_size-1)

# Subplot (b): PSD plot
# Calculate the x-values for the wavelength
xval = 112 * 4.4 / 2 / np.arange(1, 57)

# Plot the target data with a thick line
ax2.plot(xval, np.diag(np.log10(fft_mean(BARRAC2_pr.compute()))[56:, 56:]), label='BARRA-C2', linewidth=linewidth+1, color='black')

# Plot the unet data with a thick line
ax2.plot(xval, np.diag(np.log10(fft_mean(Unet_pr.compute()))[56:, 56:]), label='UNET', linewidth=linewidth+1, color='tab:blue')

# Plot the gan data with a thick line
ax2.plot(xval, np.diag(np.log10(fft_mean(GAN_pr.compute()))[56:, 56:]), label='R3GAN', linewidth=linewidth+1, color='tab:orange')

# Invert x-axis and apply log scale for x-axis
ax2.invert_xaxis()
ax2.set_xscale('log', base=2)

# Set labels and ticks for the PSD plot
ax2.set_xlabel(r'Wavelength (λ/2) [km]', fontsize=font_size)
ax2.set_ylabel('Log10(PSD)', fontsize=font_size)

# Custom x-ticks
xticks = [112 * 4.4 / 2 / i for i in [1, 2, 4, 8, 16, 32, 56]]  # Example x-tick locations
ax2.set_xticks(xticks)
ax2.set_xticklabels([f'{xt:.1f}' for xt in xticks])

# Add a legend and title for the PSD plot
ax2.legend(fontsize=font_size)
ax2.set_title('(b) Power Spectral Density (PSD)', fontsize=font_size+2)

# Increase tick font size
ax2.tick_params(axis='both', labelsize=font_size-1)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('Dist_PSD_deterministic_comp.png', dpi=300, bbox_inches="tight")
#-------------------------------------------------------------
# Deterministic spatial plot
#------------------------------------------------------------- 

# Define the time range
'''
hours = ['2021-05-25T05', '2021-05-25T06', 
         '2021-05-25T07', '2021-05-25T08', 
         '2021-05-25T09', '2021-05-25T10']       
'''
hours = ['2022-10-13T07', '2022-10-13T08', 
         '2022-10-13T09', '2022-10-13T10', 
         '2022-10-13T11', '2022-10-13T12']
# Define colors to match your colorbar image
colors = [
    (1, 1, 1),           # White (0.01 - 0.5)
    (0.17, 0.68, 0.82),  # Cyan
    (0.09, 0.49, 0.76),  # Light blue
    (0, 0.27, 0.70),     # Blue
    (0.12, 0.56, 0.22),  # Light green
    (0.17, 0.70, 0.29),  # Green
    (0.99, 0.92, 0.25),  # Yellow
    (0.97, 0.62, 0.20),  # Orange
    (0.89, 0.07, 0.19),  # Red
    (0.54, 0, 0),        # Dark red
    (0.54, 0.27, 0.07)   # Brown
]

# Compute vmax dynamically from target data
vmax = max([BARRAC2_pr.sel(time=hour).compute().max().values for hour in hours])

# Update clevels to include 0.5 as a break point
clevels = np.concatenate([np.linspace(-0.0001, 0.5, 1), np.linspace(0.5, vmax, 10)])
# Create the colormap and normalization
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(clevels)-1)
norm = mpl.colors.BoundaryNorm(clevels, cmap.N, clip=True)

# Create the figure with subplots
fig, axs = plt.subplots(len(hours), 4, figsize=(6, 8), 
                         subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

fontsize = 10
for i, hour in enumerate(hours):
    era5_data = era5_input.sel(time=hour)
    target_data = BARRAC2_pr.sel(time=hour)
    unet_data = Unet_pr.sel(time=hour)
    gan_data = GAN_pr.sel(time=hour)
    
    computed_data = dask.compute(era5_data, unet_data, gan_data, target_data)

    # Plot ERA5
    plot1 = computed_data[0].plot(ax=axs[i, 0], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 0].set_title(f"ERA5 \n{hour}", fontsize=fontsize)
    else:
        axs[i, 0].set_title(f"{hour}", fontsize=fontsize)
        
    # Plot UNET
    plot2 = computed_data[1].plot(ax=axs[i, 1], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 1].set_title(f"UNET", fontsize=fontsize)
    else:
        axs[i, 1].set_title(f"", fontsize=fontsize)
        
    # Plot GAN
    plot3 = computed_data[2].plot(ax=axs[i, 2], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 2].set_title(f"R3GAN", fontsize=fontsize)
    else:
        axs[i, 2].set_title(f"", fontsize=fontsize)

    # Plot BARRA-C2
    plot4 = computed_data[-1].plot(ax=axs[i, 3], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), add_colorbar=False)
    if i == 0:
        axs[i, 3].set_title("BARRA-C2", fontsize=fontsize)
    else:
        axs[i, 3].set_title("")

    # Add coastlines
    for ax in axs[i]:
        ax.coastlines(resolution='10m', color='black', linewidth=1)

# Add a single colorbar with the correct limits
cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=axs, orientation='vertical', fraction=0.03, pad=0.01, shrink=0.75, format=tick.FormatStrFormatter('%.2f'))
cbar.set_label('Precipitation (mm/hr)', fontsize=12)
# Increase the size of the colorbar tick labels
cbar.ax.tick_params(labelsize=8)
plt.savefig('SP_EVENT-2_deterministic_comp.png')
#-------------------------------------------------------------
# Domain plot
#-------------------------------------------------------------

# Load BARRA-C2 orography data
orog = xr.open_dataset('orog_AUST-04_ERA5_historical_hres_BOM_BARRA-C2_v1.nc')
oro = orog['orog']

# Predictor and Target domains
predictor_extent = [142, 151.25, -46, -36.75]  # [lon_min, lon_max, lat_min, lat_max]
target_extent = [144.3, 148.75, -44.25, -39.8]  # [lon_min, lon_max, lat_min, lat_max]

# Create figure and axis with Cartopy projection
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot the orography data
im = oro.sortby('lat', ascending=False).sel(lat=slice(-36.25, -46), lon=slice(141, 151.25)).plot(
    ax=ax, cmap='terrain', vmin=0, add_colorbar=False)

# Add colorbar with a flat lower edge
cbar = plt.colorbar(im, ax=ax, pad=0.04, aspect=50)
cbar.ax.tick_params(bottom=True, direction='out')
cbar.ax.set_xlabel('m', fontsize = 12)
cbar.ax.xaxis.set_label_position('top')

# Add Cartopy features
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale='50m', facecolor='white')
ax.add_feature(ocean)

# Draw predictor domain
ax.plot([predictor_extent[0], predictor_extent[1], predictor_extent[1], predictor_extent[0], predictor_extent[0]],
        [predictor_extent[2], predictor_extent[2], predictor_extent[3], predictor_extent[3], predictor_extent[2]],
        transform=ccrs.PlateCarree(), color='blue', linewidth=2, label='Predictor domain')
ax.text(predictor_extent[0] + 0.2, predictor_extent[3] - 9, 'Predictor domain',
        color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Draw target domain
ax.plot([target_extent[0], target_extent[1], target_extent[1], target_extent[0], target_extent[0]],
        [target_extent[2], target_extent[2], target_extent[3], target_extent[3], target_extent[2]],
        transform=ccrs.PlateCarree(), color='red', linewidth=2, label='Target domain')
ax.text(target_extent[0] + 0.9, target_extent[3] - 0.5, 'Target domain',
        color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Add gridlines
gridlines = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gridlines.top_labels = False  # Hide longitude labels on top
gridlines.right_labels = False  # Hide latitude labels on right

# Show the plot
plt.title("BARRA-C2 orography with predictor and target domains")
plt.savefig('Domain_plot.png')

