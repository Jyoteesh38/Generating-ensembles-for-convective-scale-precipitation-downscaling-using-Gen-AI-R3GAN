#----------------------------------------
# Import packages
#----------------------------------------
import sys
import os
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"
import time
import socket
import math
import horovod.tensorflow as hvd
import tensorflow as tf
import xarray as xr
import numpy as np
import dask
import dask.array as da
import zarr as zr
import pickle
import tensorflow as tf
print(tf.version)
from tensorflow import keras

from models import build_generator, build_discriminator
from losses import R_Dis_loss, R_Gen_loss, MSE
from utils import hvd_data_split_idx, shuffle_data_indices, batch_data_load, setup_horovod, setup_gpu

# Set the logging level to suppress warnings
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

# Initial Setup
setup_horovod()
setup_gpu()
#------------------------------------------------------------- 
# load input data from zarr
#-------------------------------------------------------------
era5 = xr.open_zarr('ERA5_ML_data.zarr')
x_tr = era5['ERA5'].sel(latitude=slice(-36.75, -46), longitude=slice(142, 151.25)).sel(time=slice('1980-01-01T00:00:00.000000000', 
                          '2020-12-31T23:00:00.000000000'))
x_v = era5['ERA5'].sel(latitude=slice(-36.75, -46), longitude=slice(142, 151.25)).sel(time=slice('2021-01-01T00:00:00.000000000', 
                          '2022-12-31T23:00:00.000000000'))
# Load the static input data from zarr
x_static = xr.open_zarr('B_C2_static.zarr')['BC2_static']

b2c = xr.open_zarr('B2C_Pr_data_2k25.zarr')

# Batch size - aim to fill GPU memory to achieve best computational performance
batch_size = 32

Epoch_size = x_tr.shape[0]
Val_size = x_v.shape[0]
istart, istop = hvd_data_split_idx(hvd.rank(), Epoch_size, hvd.size())
i_val_start, i_val_stop = hvd_data_split_idx(hvd.rank(), Val_size, hvd.size())

train_indices = shuffle_data_indices(list(range(Epoch_size)), 0)[istart:istop]
val_indices = shuffle_data_indices(list(range(Val_size)), 0)[i_val_start:i_val_stop]

x_train = x_tr[train_indices,...]
y_train = b2c['pr'].sel(time=slice('1980-01-01T00:00:00.000000000', 
                          '2020-12-31T23:30:00.000000000')).expand_dims(channel=1, axis=-1)[train_indices,...]
x_val = x_v[val_indices,...]
y_val = b2c['pr'].sel(time=slice('2021-01-01T00:00:00.000000000', 
                          '2022-12-31T23:30:00.000000000')).expand_dims(channel=1, axis=-1)[val_indices,...]

print('*** rank = ', hvd.rank(),' Training data shapes = ', x_train.shape, y_train.shape)
print('*** rank = ', hvd.rank(),' Val data shapes = ', x_val.shape, y_val.shape)

# Determine how many batches are there in train and val sets
train_batches = int(math.floor(len(train_indices) / batch_size))
val_batches = int(math.floor(len(val_indices) / batch_size))

print ('*** rank = ', hvd.rank(),' train_batches', train_batches)
print ('*** rank = ', hvd.rank(),' val_batches', val_batches)

# -------------------------------------------------------------------------
# Model initialising and summary print 
# -------------------------------------------------------------------------
# Build the generator model
generator = build_generator(
    inp_lat=x_train.shape[1], inp_lon=x_train.shape[2],
    out_lat=y_train.shape[1], out_lon=y_train.shape[2],
    chnl=x_train.shape[3], out_vars=y_train.shape[3],
    fil=5, dil_rate=1, std=1, swtch=1, alpha_val=0.2,
    reg_val=0.0, num_heads=1, key_dim=64
)
# Build the discriminator model
discriminator = build_discriminator(
    inp_lat=x_train.shape[1], inp_lon=x_train.shape[2],
    out_lat=y_train.shape[1], out_lon=y_train.shape[2],
    chnl=x_train.shape[3], out_vars=y_train.shape[3],
    n_out=64, fil=5, dil_rate=1, std=1,
    alpha_val=0.2, reg_val=0.0
)

g_opt = tf.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
d_opt = tf.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

print(generator.summary()) if hvd.rank() == 1 else None
print(discriminator.summary()) if hvd.rank() == 1 else None

#---------------------------------------------------------------------------
# Train the model
#---------------------------------------------------------------------------
#--------------------------------------------------------------------
# Define first training step (defined seperately to effectively utilise tf.function),
# training step and validation step
#--------------------------------------------------------------------
@tf.function
def first_training_step(x, y_st, y):

    y_ns = tf.random.normal(shape=[y.shape[0], y.shape[1], y.shape[2], 64], dtype=y.dtype)
    #Train the discriminator three times as frequently as the generator
    for i in range(3):
        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Forward pass
            y_fake = generator([x, y_st, y_ns], training=True)
            yd_fake = discriminator([x, y_st, y_fake], training=True)
            yd_real = discriminator([x, y_st, y], training=True)
            # Compute Relativistic Discriminator loss with zero gradient penalties
            D_loss, R1, R2 = R_Dis_loss(x, y_st, y, y_fake, yd_real, yd_fake, 1)

        # Horovod: add Horovod Distributed GradientTape
        d_tape = hvd.DistributedGradientTape(d_tape)
        # Compute gradients
        d_gradients = d_tape.gradient(D_loss, discriminator.trainable_variables)
        # Update weights
        d_opt.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        if i == 0:

            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.

            hvd.broadcast_variables(discriminator.variables, root_rank=0)
            hvd.broadcast_variables(d_opt.variables(), root_rank=0)

    # Train the generator
    with tf.GradientTape() as g_tape:

        # Forward pass
        y_pr = generator([x, y_st, y_ns], training=True)
        yg_fake = discriminator([x, y_st, y_pr], training=True)
        yg_real = discriminator([x, y_st, y], training=True)
        MSE_loss = MSE(y, y_pr)
        # Compute total Generator loss = sum of mse and relativistic generator loss
        G_loss = (MSE_loss) + (0.009 *  R_Gen_loss(yg_real, yg_fake))

    # Horovod: add Horovod Distributed GradientTape
    g_tape = hvd.DistributedGradientTape(g_tape)
    # Compute gradients
    g_gradients = g_tape.gradient(G_loss, generator.trainable_variables)
    # Update weights
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    hvd.broadcast_variables(generator.variables, root_rank=0)
    hvd.broadcast_variables(g_opt.variables(), root_rank=0)

    return D_loss, R1, R2, G_loss, MSE_loss 

@tf.function
def training_step(x, y_st, y):
    y_ns = tf.random.normal(shape=[y.shape[0], y.shape[1], y.shape[2], 64], dtype=y.dtype)
    #Train the discriminator three times as frequently as the generator
    for i in range(3):
        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Forward pass
            y_fake = generator([x, y_st, y_ns], training=True)
            yd_fake = discriminator([x, y_st, y_fake], training=True)
            yd_real = discriminator([x, y_st, y], training=True)
            # Compute Relativistic Discriminator loss with zero gradient penalties
            D_loss, R1, R2 = R_Dis_loss(x, y_st, y, y_fake, yd_real, yd_fake, 1)

        # Horovod: add Horovod Distributed GradientTape
        d_tape = hvd.DistributedGradientTape(d_tape)
        # Compute gradients
        d_gradients = d_tape.gradient(D_loss, discriminator.trainable_variables)
        # Update weights
        d_opt.apply_gradients(zip(d_gradients, discriminator.trainable_variables))


    # Train the generator
    with tf.GradientTape() as g_tape:

        # Forward pass
        y_pr = generator([x, y_st, y_ns], training=True)
        yg_fake = discriminator([x, y_st, y_pr], training=True)
        yg_real = discriminator([x, y_st, y], training=True)
        MSE_loss = MSE(y, y_pr)
        # Compute total Generator loss = sum of mse and relativistic generator loss
        G_loss = (MSE_loss) + (0.009 *  R_Gen_loss(yg_real, yg_fake))

    # Horovod: add Horovod Distributed GradientTape
    g_tape = hvd.DistributedGradientTape(g_tape)
    # Compute gradients
    g_gradients = g_tape.gradient(G_loss, generator.trainable_variables)
    # Update weights
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return D_loss, R1, R2, G_loss, MSE_loss 

@tf.function
def validation_step(x, y_st, y, y_ns):
    y_pr = generator([x, y_st, y_ns], training=False)
    yg_fake = discriminator([x, y_st, y_pr], training=False)
    yg_real = discriminator([x, y_st, y], training=False)
    val_mse = MSE(y, y_pr)
    val_G_loss = val_mse + (0.009 *  R_Gen_loss(yg_real, yg_fake))
    val_D_loss, val_R1, val_R2 = R_Dis_loss(x, y_st, y, y_pr, yg_real, yg_fake, 1)
    return val_G_loss, val_D_loss, val_mse, val_R1, val_R2
#---------------------------------------------------------------------------
# Defining validation step to monitor train and val loss at each epoch
#@tf.function
def val_step(x, x_s, y, batches, batch_size):
    kstart = 0
    total_samples = 0
    outputs_agg = None  # Initialize outputs_agg as None
    for batch in range(batches):
        kend = kstart + batch_size
        if kend >= len(x):
            kend = len(x)
        x_in = batch_data_load(x, kstart, kend)
        y_out = batch_data_load(y, kstart, kend)
        xs_in = batch_data_load(x_s, 0, batch_size)

        y_ns = tf.random.normal(shape=[y_out.shape[0], y_out.shape[1], y_out.shape[2], 64], dtype=y_out.dtype)
        # Perform validation step
        outputs = validation_step(x_in, xs_in, y_out[..., tf.newaxis], y_ns)
        # Initialize outputs_agg if it's None
        if outputs_agg is None:
            outputs_agg = outputs  # Initialize outputs_agg with the first batch
        else:
        # Accumulate outputs with each batch
            outputs_agg = [agg + output for agg, output in zip(outputs_agg, outputs)]
        kstart = kend

    # Calculate average of accumulated outputs across all batches
    outputs_agg = [output / batches for output in outputs_agg]
    #outputs_agg = outputs_agg / batches
    # Combine the variables into a single tensor and reduce across ranks
    Metrics = tf.convert_to_tensor(outputs_agg, dtype=tf.float32)
    #print('*** rank = ', hvd.rank(), "I am just here before all reduce")
    Metrics_val = hvd.allreduce(Metrics, average=True)
    return Metrics_val

# ------------------------------------------------------------------------
# Add a barrier to sync all processes before starting training
barrier = hvd.allreduce(tf.constant(0))
print ('*** rank = ', hvd.rank(),' Train model')
#----------------------------------------------------------------
# Running epochs
#----------------------------------------------------------------
max_epochs = 200
previous_epoch = 0
for epoch in range(previous_epoch, max_epochs):
    epoch_start = time.time()
    lr = 1e-5
    if hvd.rank() == 2: 
        print(f"Epoch {epoch+1}: Learning Rate = {lr}")
    g_opt.learning_rate.assign(lr)
    d_opt.learning_rate.assign(lr)
    #------------------------------------------------------------------------------------------------
    # Horovod: shuffle the training and val data across multiple processors
    # Set the same seed for all GPUs to shuffle data consistently
    # Randomising the train and val data in each epoch and training and testing on 200 and 60 batches
    # -----------------------------------------------------------------------------------------------
    train_indices = shuffle_data_indices(list(range(Epoch_size)), epoch)[istart:istop]
    val_indices = shuffle_data_indices(list(range(Val_size)), epoch)[i_val_start:i_val_stop]

    y_train = b2c['pr'].sel(time=slice('1980-01-01T00:00:00.000000000', 
                          '2020-12-31T23:30:00.000000000'))[train_indices,...]
    y_val = b2c['pr'].sel(time=slice('2021-01-01T00:00:00.000000000', 
                          '2022-12-31T23:30:00.000000000'))[val_indices,...]
    x_train = x_tr[train_indices,...]
    x_val = x_v[val_indices,...]
    #-----------------------------------
    # Executing training step on batches
    # ----------------------------------
    jstart = 0
    for batch in range(train_batches):
        jend = jstart + batch_size
        if jend >= len(train_indices):
            jend = len(train_indices)

        x_in = batch_data_load(x_train, jstart, jend)
        y_out = batch_data_load(y_train, jstart, jend)
        xs_in = batch_data_load(x_static, 0, batch_size)

        if ((epoch == 0) and (batch == 0)):
            d_batch_loss, R1_batch_loss, R2_batch_loss, g_batch_loss, mse_batch_loss = first_training_step(x_in, xs_in, y_out[..., tf.newaxis])
        else:
            d_batch_loss, R1_batch_loss, R2_batch_loss, g_batch_loss, mse_batch_loss = training_step(x_in, xs_in, y_out[..., tf.newaxis])

        jstart = jend

        if ((hvd.rank() == 1) & (batch % 5 == 0)):
            print(f"rank:{hvd.rank()}, epoch:{epoch+1}, batch no:{batch}, dis_batch_loss:{d_batch_loss}, R1_batch_loss:{R1_batch_loss}, R2_batch_loss:{R2_batch_loss}, gen_batch_loss:{g_batch_loss}, gen_mse_batch_loss:{mse_batch_loss}")

    epoch_train_end = time.time()
    if hvd.rank() == 2:
        print('*** rank = ', hvd.rank(), 'Lap time for epoch - training: ', epoch_train_end - epoch_start)
    #-------------------------------------
    # Executing validation step on batches
    # ------------------------------------
    # Train loss values
    train_loss_metrics = val_step(x_train, x_static, y_train, train_batches, batch_size)
    # Val loss values
    val_loss_metrics = val_step(x_val, x_static, y_val, val_batches, batch_size)
    #---------------------------------------
    epoch_end = time.time()
    if hvd.rank() == 3:
        print('*** rank = ', hvd.rank(), 'Lap time for epoch - training and validation: ',epoch_end - epoch_start)
    if hvd.rank() == 0:
        print("End of epoch: ", epoch+1, "lr :", g_opt.lr.numpy())
        print("G_Loss: ", train_loss_metrics[0].numpy(), "D_Loss: ", train_loss_metrics[1].numpy(), "MSE:", train_loss_metrics[2].numpy(), "R1:", train_loss_metrics[3].numpy(), "R2:", train_loss_metrics[4].numpy())
        print("Val_G_loss: ", val_loss_metrics[0].numpy(), "Val_D_loss: ", val_loss_metrics[1].numpy(), "Val_MSE:", val_loss_metrics[2].numpy(), "Val_R1:", val_loss_metrics[3].numpy(), "Val_R2:", val_loss_metrics[4].numpy())
        # Specify the file path where you want to save the learning rates
        file_path = "history.txt"
        if epoch == 0:
            # Open the file in write mode
            file = open(file_path, "w")
            line = f"Epoch,G_Loss,D_Loss,MSE,R1,R2,Val_G_loss,Val_D_loss,Val_MSE,Val_R1,Val_R2 \n"
            file.write(line)
            line = f"{epoch+1},{train_loss_metrics[0].numpy()},{train_loss_metrics[1].numpy()},{train_loss_metrics[2].numpy()},{train_loss_metrics[3].numpy()},{train_loss_metrics[4].numpy()},{val_loss_metrics[0].numpy()},{val_loss_metrics[1].numpy()},{val_loss_metrics[2].numpy()},{val_loss_metrics[3].numpy()},{val_loss_metrics[4].numpy()} \n"
            file.write(line)
            file.close()
        else:
            file = open(file_path, "a") 
            line = f"{epoch+1},{train_loss_metrics[0].numpy()},{train_loss_metrics[1].numpy()},{train_loss_metrics[2].numpy()},{train_loss_metrics[3].numpy()},{train_loss_metrics[4].numpy()},{val_loss_metrics[0].numpy()},{val_loss_metrics[1].numpy()},{val_loss_metrics[2].numpy()},{val_loss_metrics[3].numpy()},{val_loss_metrics[4].numpy()} \n"
            file.write(line)
            file.close()
        generator.save('Gen_R3GAN_pr_{}.h5'.format(epoch+1))
        discriminator.save('Dis_R3GAN_pr_{}.h5'.format(epoch+1))

