import xarray as xr
import matplotlib.pyplot as plt

from ensemble_predictions import predict_ensemble

#------------------------------------------------------------- 
# load input and target data
#-------------------------------------------------------------
era5 = xr.open_zarr('ERA5_ML_data_2k25_nested_directory.zarr')
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

# Run the ensemble prediction function
ens_pred, target = predict_ensemble(n_steps=17520, mask=None)

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
