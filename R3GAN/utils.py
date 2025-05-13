import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd

def setup_horovod():
    hvd.init()
    print(f"[Rank {hvd.rank()}] Horovod initialized with {hvd.size()} workers")

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#----------------------------------------------------------------
# Horovod: Split the training and val data across multiple processors   
#----------------------------------------------------------------
def hvd_data_split_idx(rank, sample_size, size):
    istart = int(rank*sample_size/size)
    istop  = int((rank+1)*sample_size/size)
    return istart, istop

#----------------------------------------------------------------
# Horovod: shuffle the training and val data across multiple processors   
#----------------------------------------------------------------
# Set the same seed for all GPUs to shuffle data consistently
seed = 1
# Function to shuffle data indices
def shuffle_data_indices(indices, seed):
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices

# --------------------------------------------------------------
# Batches data load
# --------------------------------------------------------------
def batch_data_load(var, start, end):
    return tf.convert_to_tensor(var[start:end,...].compute().values, dtype=tf.float32)
