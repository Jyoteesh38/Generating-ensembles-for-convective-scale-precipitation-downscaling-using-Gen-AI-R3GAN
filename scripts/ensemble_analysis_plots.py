
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
