import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import get_worker
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.utils import custom_object_scope
import os
import random
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"

#------------------------------------------------------------- 
# Denormalise the predicted and target pr data
#-------------------------------------------------------------
def denorm(x):
    max_val = 110.86449623107906
    out = (10**(x * np.log10(1 + max_val))) - 1
    # Thresholding: set all values less than 0.25 mm/hr to zero
    out = out.where(out >= 0.25, 0)
    return out

def predict_ensemble(n_steps, mask=None):
    """
    Predict using an ensemble of GAN models with different random seeds and varying stddev values.

    :param n_steps: Number of time steps for prediction.
    :param mask: Optional mask to apply.
    """
    
    class ReflectPadding2D(Layer):
        def __init__(self, pad_size, **kwargs):
            self.pad_size = pad_size
            super(ReflectPadding2D, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], mode='REFLECT')

        def get_config(self):
            config = super(ReflectPadding2D, self).get_config()
            config.update({"pad_size": self.pad_size})
            return config
    
    # Load both models
    model_path_1 = "Gen_R3GAN_pr_115.h5"
    model_path_2 = "Gen_R3GAN_pr_121.h5"
    model_path_3 = "Gen_R3GAN_pr_123.h5"
    model_path_4 = "Gen_R3GAN_pr_125.h5"
    model_path_5 = "Gen_R3GAN_pr_129.h5"
    
    model_1 = tf.keras.models.load_model(model_path_1, custom_objects={'ReflectPadding2D': ReflectPadding2D}, compile=False)
    model_2 = tf.keras.models.load_model(model_path_2, custom_objects={'ReflectPadding2D': ReflectPadding2D}, compile=False)
    model_3 = tf.keras.models.load_model(model_path_3, custom_objects={'ReflectPadding2D': ReflectPadding2D}, compile=False)
    model_4 = tf.keras.models.load_model(model_path_4, custom_objects={'ReflectPadding2D': ReflectPadding2D}, compile=False)
    model_5 = tf.keras.models.load_model(model_path_5, custom_objects={'ReflectPadding2D': ReflectPadding2D}, compile=False)
    
    t_start = 0
    t_end = t_start + n_steps
    
    def generate_prediction(model, seed, stddev):
        """Generate a single GAN ensemble member prediction."""
        tf.random.set_seed(seed)
        noise = tf.random.normal(shape=[n_steps, 112, 112, 64], stddev=stddev, seed=seed, dtype='float32')
        y_pred = model.predict([e5[t_start:t_end, ...].compute().values, 
                                x_static[t_start:t_end, ...].compute().values, 
                                noise])
        
        # Convert to xarray.DataArray
        y_pred_da = xr.DataArray(y_pred, 
                                 coords=[b2c_pr.coords['time'][t_start:t_end], b2c_pr.coords['lat'], b2c_pr.coords['lon'], [0]], 
                                 dims=['time', 'lat', 'lon', 'member'])
        return denorm(y_pred_da[..., 0])

    # Generate ensemble members
    delayed_predictions = []
    # Generate 10 ensemble members from four models each with 25 members
    for i in range(0, 2):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_1, i, 1.1))
        
    for i in range(0, 2):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_2, i, 1.0))
        
    for i in range(0, 2):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_3, i, 1.0))
        
    for i in range(0, 2):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_4, i, 1.0))
        
    for i in range(0, 2):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_5, i, 1.15))
    '''    
    # Generate ensemble members
    stddev_values = np.linspace(0.75, 1.2, 10)  # Varying stddev from 0.75 to 1.15
    # Next 50 members: Fix the seed but vary stddev
    for i, stddev in enumerate(stddev_values):
        delayed_predictions.append(dask.delayed(generate_prediction)(model_5, i, 1.15))
    ''' 
    # Compute ensemble predictions in parallel
    with ProgressBar():
        ensemble_predictions = dask.compute(*delayed_predictions)

    # Stack ensemble predictions along a new ensemble dimension
    ensemble_predictions = xr.concat(ensemble_predictions, dim='member').transpose('time', 'lat', 'lon', 'member')

    # Convert to Dask arrays for efficient computation
    ensemble_predictions = ensemble_predictions.chunk({'time': -1, 'member': -1, 'lat': 112, 'lon': 112})
    
    # True target data
    target_data = denorm(b2c_pr[t_start:t_end, ..., 0])
    
    # Check shapes
    print(f"Shape of ensemble_predictions: {ensemble_predictions.shape}")
    print(f"Shape of target_data: {target_data.shape}")

    # Check value ranges
    print(f"Min/Max of ensemble_predictions: {ensemble_predictions.min().values}, {ensemble_predictions.max().values}")
    print(f"Min/Max of target_data: {target_data.min().values}, {target_data.max().values}")
    
    return ensemble_predictions, target_data
