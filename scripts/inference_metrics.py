import numpy as np
import xskillscore as xs
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def fft_mean(rx):
    
    # Initialize the array to accumulate FFT results
    np_fft = np.zeros_like(rx[0, :, :])
    
    iend = int(len(rx[:, 0, 0]))
    
    # Loop through each slice and compute FFT
    for i in range(0, iend):
        x = rx[i, :, :]
        fft = np.square(np.absolute(np.fft.fftshift(np.fft.fft2(x))))
        np_fft += fft
    
    # Average the FFT results
    y = np_fft / iend
    
    # Free up memory
    del np_fft
    return y

# calculate_normalized_rank_histogram with xskillscore package
def calculate_normalized_rank_histogram_xs(ensemble_predictions, target_data):
    """
    Calculate the normalized rank histogram using xskillscore.rank_histogram.
    
    Parameters:
        ensemble_predictions (xarray.DataArray): Ensemble forecasts with dimensions (time, lat, lon, member).
        target_data (xarray.DataArray): Observations with dimensions (time, lat, lon).
    
    Returns:
        np.ndarray: Normalized rank histogram.
    """
    
    # Add small noise to avoid ties
    noise_level = 1e-6
    ensemble_predictions += np.random.normal(0, noise_level, ensemble_predictions.shape)
    target_data += np.random.normal(0, noise_level, target_data.shape)
    
    # Compute rank histogram using xskillscore
    rank_hist = xs.rank_histogram(target_data, ensemble_predictions, member_dim="member")
    
    # Convert rank histogram to numpy array
    rank_hist_np = rank_hist.values

    # Normalize histogram to have area sum to 1
    rank_hist_np = rank_hist_np / np.sum(rank_hist_np)
    
    return rank_hist_np

def plot_normalized_rank_histogram_xs(norm_rank_hist):
    """
    Plot the normalized rank histogram with normalized occurrence on the y-axis as a line plot,
    and overlay the ideal rank histogram as a dashed line.
    
    Parameters:
        norm_rank_hist (np.ndarray): Normalized rank histogram values.
    """
    bins = np.linspace(0, 1, len(norm_rank_hist) + 1)
    #bins = np.linspace(0, 1, 10 + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]  # Calculate bin width
    
    # Compute ideal rank histogram (uniform distribution)
    ideal_hist = np.ones_like(norm_rank_hist) / len(norm_rank_hist)
    
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, norm_rank_hist, width=bin_width, edgecolor="black", alpha=0.7, label="Ensemble-3 rank histogram")
    plt.plot(bin_centers, ideal_hist, linestyle='--', color='red', label="Ideal rank histogram")
    
    font_size = 14
    plt.xlabel("Normalized rank", fontsize=font_size)
    plt.ylabel("Normalized occurrence", fontsize=font_size)
    plt.title("(c) Ensemble-3", fontsize=font_size+2)
    plt.legend(fontsize=font_size)
    
    # Increase tick font size
    plt.tick_params(axis='both', labelsize=font_size-1)
    plt.grid(True, linestyle="--", alpha=0.6)
    # Ensure y-axis starts from zero
    plt.ylim(bottom=0)
    plt.savefig('RK_hist_R3GAN8_ens_3.pdf')

def calculate_probabilities(ensemble_predictions, threshold):
    """
    Calculate the probability that the precipitation exceeds the threshold
    based on the fraction of ensemble members predicting above the threshold.
    
    :param ensemble_predictions: Ensemble predictions (xarray DataArray with 'member' dimension).
    :param threshold: Precipitation threshold (e.g., 0.5 mm/hr, 5.0 mm/hr).
    
    :return: Probabilities for each pixel.
    """
    # Calculate the fraction of ensemble members that predict above the threshold
    exceedance = (ensemble_predictions >= threshold).astype(int)
    probabilities = exceedance.mean(dim='member')
    
    return probabilities

def calculate_roc_for_ensemble(ensemble_predictions, target_data, threshold):
    """
    Calculate the ROC curve for the ensemble predictions against the target data.
    
    :param ensemble_predictions: Ensemble predictions (xarray DataArray with 'member' dimension).
    :param target_data: True target data (xarray DataArray).
    :param threshold: Threshold for binarizing precipitation values (e.g., 0.5 mm/hr, 5 mm/hr).
    
    :return: false positive rates, true positive rates, and ROC AUC score.
    """
    # Calculate probabilities based on ensemble exceedance
    probabilities = calculate_probabilities(ensemble_predictions, threshold)
    
    # Binarize the target data based on the threshold
    target_binary = (target_data >= threshold).astype(int)
    
    # Flatten the arrays for roc_curve calculation
    y_true = target_binary.values.flatten()
    y_scores = probabilities.values.flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curves(thresholds, ensemble_predictions, target_data):
    """
    Plot ROC curves for multiple thresholds in a single figure with subplots.
    
    :param thresholds: List of thresholds to evaluate.
    :param ensemble_predictions: Ensemble predictions (xarray DataArray with 'member' dimension).
    :param target_data: True target data (xarray DataArray).
    """
    fig, axes = plt.subplots(1, len(thresholds), figsize=(12, 6))

    for i, threshold in enumerate(thresholds):
        fpr, tpr, roc_auc = calculate_roc_for_ensemble(ensemble_predictions, target_data, threshold)

        axes[i].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes[i].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Diagonal line
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        font_size = 14
        axes[i].set_xlabel('False positive rate', fontsize=font_size)
        axes[i].set_ylabel('True positive rate', fontsize=font_size)
        axes[i].set_title(f'({chr(97 + i + 4)}) ROC curve (Threshold = {threshold} mm/hr)', fontsize=font_size)
        axes[i].legend(loc="lower right", fontsize=font_size)
        # Increase tick font size
        axes[i].tick_params(axis='both', labelsize=font_size-1)

    # Add main title
    fig.suptitle("Ensemble-3", fontsize=font_size+2)
    plt.tight_layout()
    plt.savefig('ROC_R3GAN8_ens_3.pdf')
