import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, classification_report
from sklearn.calibration import calibration_curve
import seaborn as sns
from ad import utils

def plot_histograms(data, bins=100, range=(0, 1000), ylog=False, cut=None, title=None, hnormalize=False, figsize=(6, 3)):
    # Filter the data based on the condition (value > 10)
    filt = np.where(data > 10, data, 0)
    M1 = np.sum(filt[:, 0:152:8], axis=1)
    M2 = np.sum(filt[:, 4:152:8], axis=1)

    print(f'The M1 array is {M1}')
    print(f'The M2 array is {M2}')

    # Plot the histograms
    plt.figure(figsize=figsize)

    # Plot M1 histogram
    plt.subplot(1, 2, 1)
    plt.hist(M1, bins=bins, color='blue', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('SUM QM1[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M1' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    if cut != None:
        plt.axvline(x=cut, color='r', linestyle='--',label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('SUM QM2[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M2' or 'Histogram of M2')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    
    if cut != None:
        plt.axvline(x=cut, color='b', linestyle='--', label='axvline2 - full height')
    

    plt.tight_layout()
    plt.show()

    return M1, M2

def cut_data_by_threshold(data, cut, bins=150, range=(0, 1000), ylog=False, hnormalize=False, title=None, figsize=(6, 3)):

    filt = np.where(data > 10, data, 0)

    M1 = np.sum(filt[:, 0:152:8], axis=1)
    M2 = np.sum(filt[:, 4:152:8], axis=1)
    del filt
    utils.free_mem()

    # Filter the data based on the condition (M1 > cut) and (M2 > cut)
    filtered_data = data[(M1 >= cut) & (M2 >= cut)]
    #print(f'data{filtered_data}')
    print(f"The final shape of {title} data is {filtered_data.shape}")
    
    filt = np.where(filtered_data > 10, filtered_data, 0)
    M1 = np.sum(filt[:, 0:152:8], axis=1)
    M2 = np.sum(filt[:, 4:152:8], axis=1)

    # Plot the histograms
    plt.figure(figsize=figsize)

    # Plot M1 histogram
    plt.subplot(1, 2, 1)
    plt.hist(M1, bins=bins, color='blue', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('SUM QM1[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M1' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    plt.axvline(x=cut, color='r', linestyle='--',label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('SUM QM2[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M2' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    plt.axvline(x=cut, color='b', linestyle='--',label='axvline2 - full height')

    plt.tight_layout()
    plt.show()

    return filtered_data

def plot_normalized_histogram(predictions, bins = 120, prob_cut=None):
    # Split predictions into three separate arrays for each class
    p_predictions = predictions[:, 0]
    he_predictions = predictions[:, 1]
    fe_predictions = predictions[:, 2]


    # Create histogram for each class
    p_counts, p_bins, _ = plt.hist(p_predictions, bins=bins, alpha=0, density=True)
    he_counts, he_bins, _ = plt.hist(he_predictions, bins=bins, alpha=0, density=True)
    fe_counts, fe_bins, _ = plt.hist(fe_predictions, bins=bins, alpha=0, density=True)

    # Normalize the distributions
    p_counts /= np.max(p_counts)
    he_counts /= np.max(he_counts)
    fe_counts /= np.max(fe_counts)

    # Set the range of the y-axis to 0-1
    # plt.ylim(0, 1.1)

    # Plot the normalized histograms
    plt.hist(p_bins[:-1], bins=p_bins, weights=p_counts, label='p', color='blue', alpha=0.3, histtype='step', hatch='///')
    plt.hist(he_bins[:-1], bins=he_bins, weights=he_counts, label='He', color='red', alpha=0.3, histtype='step', hatch='///')
    plt.hist(fe_bins[:-1], bins=fe_bins, weights=fe_counts, label='Fe', color='green', alpha=0.3)

    # Set plot labels and legend
    plt.xlabel('Predicted Probability')
    plt.ylabel('Normalized Number of Entries')
    plt.yscale('log')
    if prob_cut is not None:
        plt.axvline(x=prob_cut, color='black', linestyle='--', label='Cut')

    plt.legend(loc='upper center')

    # Show or save the plot
    plt.show()

def plot_cumulative(predictions, bins=50, prob_cut=None):
    prob_cut = prob_cut or None

    # Split predictions into three separate arrays for each class
    p_predictions = predictions[:, 0]
    he_predictions = predictions[:, 1]
    fe_predictions = predictions[:, 2]

    # Create histogram for each class
    p_counts, p_bins, _ = plt.hist(p_predictions, bins=bins, alpha=0, density=True)
    he_counts, he_bins, _ = plt.hist(he_predictions, bins=bins, alpha=0, density=True)
    fe_counts, fe_bins, _ = plt.hist(fe_predictions, bins=bins, alpha=0, density=True)

    # Normalize the distributions
    p_counts /= np.max(p_counts)
    he_counts /= np.max(he_counts)
    fe_counts /= np.max(fe_counts)

    # Set the range of the y-axis to 0-1
    # plt.ylim(0, 1.1)

    # Plot the normalized histograms
    plt.hist(p_bins[:-1], bins=p_bins, weights=p_counts, alpha=0)
    plt.hist(he_bins[:-1], bins=he_bins, weights=he_counts, alpha=0)
    plt.hist(fe_bins[:-1], bins=fe_bins, weights=fe_counts, alpha=0)

    # Calculate cumulative histograms
    p_cumulative = np.cumsum(p_counts[::-1])[::-1]
    he_cumulative = np.cumsum(he_counts[::-1])[::-1]
    fe_cumulative = np.cumsum(fe_counts[::-1])[::-1]

    # Normalize cumulative histograms
    p_cumulative /= np.max(p_cumulative)
    he_cumulative /= np.max(he_cumulative)
    fe_cumulative /= np.max(fe_cumulative)

    # Generate cumulative histogram plot
    plt.plot(p_bins[:-1], p_cumulative, label='p (Cumulative)', color='blue', linestyle='dotted')
    plt.plot(he_bins[:-1], he_cumulative, label='He (Cumulative)', color='red', linestyle='dotted')
    plt.plot(fe_bins[:-1], fe_cumulative, label='Fe (Cumulative)', color='green', linestyle='dotted')

    # Set plot labels and legend
    plt.xlabel('Predicted Probability')
    plt.ylabel('Rejection Power')
    plt.legend(loc='lower left')
    if prob_cut is not None:
        plt.xlim(prob_cut, 1)

    # Set y-axis to log scale
    plt.yscale('log')

    # Show or save the plot
    plt.show()

def plot_compared_histograms(predictions_single, predictions_ensemble, prob_cut=None):
    # Split predictions into three separate arrays for each class
    p_predictions_single = predictions_single[:, 0]
    he_predictions_single = predictions_single[:, 1]
    fe_predictions_single = predictions_single[:, 2]
    
    p_predictions_ensemble = predictions_ensemble[:, 0]
    he_predictions_ensemble = predictions_ensemble[:, 1]
    fe_predictions_ensemble = predictions_ensemble[:, 2]

    # Create histogram for each class
    p_counts_single, p_bins_single, _ = plt.hist(p_predictions_single, bins=100, alpha=0, density=True)
    he_counts_single, he_bins_single, _ = plt.hist(he_predictions_single, bins=100, alpha=0, density=True)
    fe_counts_single, fe_bins_single, _ = plt.hist(fe_predictions_single, bins=100, alpha=0, density=True)
    
    p_counts_ensemble, p_bins_ensemble, _ = plt.hist(p_predictions_ensemble, bins=100, alpha=0, density=True)
    he_counts_ensemble, he_bins_ensemble, _ = plt.hist(he_predictions_ensemble, bins=100, alpha=0, density=True)
    fe_counts_ensemble, fe_bins_ensemble, _ = plt.hist(fe_predictions_ensemble, bins=100, alpha=0, density=True)

    # Normalize the distributions
    p_counts_single /= np.max(p_counts_single)
    he_counts_single /= np.max(he_counts_single)
    fe_counts_single /= np.max(fe_counts_single)
    
    p_counts_ensemble /= np.max(p_counts_ensemble)
    he_counts_ensemble /= np.max(he_counts_ensemble)
    fe_counts_ensemble /= np.max(fe_counts_ensemble)

    # Set colors for each class
    colors_p = ['lightblue', 'darkblue']
    colors_he = ['lightcoral', 'darkred']
    colors_fe = ['lightgreen', 'darkgreen']

    # Plot the normalized histograms
    plt.hist(p_bins_single[:-1], bins=p_bins_single, weights=p_counts_single, label='p (SDNN)', color=colors_p[0], histtype='step', linewidth=1.5)
    plt.hist(he_bins_single[:-1], bins=he_bins_single, weights=he_counts_single, label='He (SDNN)', color=colors_he[0], histtype='step', linewidth=1.5)
    plt.hist(fe_bins_single[:-1], bins=fe_bins_single, weights=fe_counts_single, label='Fe (SDNN)', color=colors_fe[0], histtype='step', linewidth=1.5)
    
    plt.hist(p_bins_ensemble[:-1], bins=p_bins_ensemble, weights=p_counts_ensemble, label='p (EDNN)', color=colors_p[1], histtype='step', linewidth=1.5)
    plt.hist(he_bins_ensemble[:-1], bins=he_bins_ensemble, weights=he_counts_ensemble, label='He (EDNN)', color=colors_he[1], histtype='step', linewidth=1.5)
    plt.hist(fe_bins_ensemble[:-1], bins=fe_bins_ensemble, weights=fe_counts_ensemble, label='Fe (EDNN)', color=colors_fe[1], histtype='step', linewidth=1.5)

    # Set plot labels and legend
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Normalized Number of Entries', fontsize=10)

    # plt.title('Score Histogram', fontsize=14)

    plt.ylim(0, 1.1)
    # plt.yscale('log')
    if prob_cut is not None:
        plt.axvline(x=prob_cut, color='black', linestyle='--', label='Cut')

    # Get handles and labels of the current legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort handles and labels based on class names
    handles = [handles[i] for i in [0, 3, 1, 4, 2, 5]]
    labels = [labels[i] for i in [0, 3, 1, 4, 2, 5]]

    # Create a new legend with sorted handles and labels
    plt.legend(handles, labels, loc='lower right', fontsize=8)

    # Remove top and right spines
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # Add gridlines
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    
    # Move the ticks to the inside of the plot
    plt.tick_params(axis='both', direction='in')

    # Add ticks on the upper and right parts of the plots
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_tick_params(which='both', top=True, bottom=True, direction='in')
    plt.gca().yaxis.set_tick_params(which='both', left=True, right=True, direction='in')
    plt.figure(figsize=(8,6))

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_rejection_power(predictions_single, predictions_ensemble, prob_cut=None, bin = 100):
    # Split predictions into three separate arrays for each class
    p_predictions_single = predictions_single[:, 0]
    he_predictions_single = predictions_single[:, 1]
    fe_predictions_single = predictions_single[:, 2]
    
    p_predictions_ensemble = predictions_ensemble[:, 0]
    he_predictions_ensemble = predictions_ensemble[:, 1]
    fe_predictions_ensemble = predictions_ensemble[:, 2]

    # Create histogram for each class
    p_counts_single, p_bins_single, _ = plt.hist(p_predictions_single, bins=bin, alpha=0, density=True)
    he_counts_single, he_bins_single, _ = plt.hist(he_predictions_single, bins=bin, alpha=0, density=True)
    fe_counts_single, fe_bins_single, _ = plt.hist(fe_predictions_single, bins=bin, alpha=0, density=True)
    
    p_counts_ensemble, p_bins_ensemble, _ = plt.hist(p_predictions_ensemble, bins=bin, alpha=0, density=True)
    he_counts_ensemble, he_bins_ensemble, _ = plt.hist(he_predictions_ensemble, bins=bin, alpha=0, density=True)
    fe_counts_ensemble, fe_bins_ensemble, _ = plt.hist(fe_predictions_ensemble, bins=bin, alpha=0, density=True)

    # Normalize the distributions
    p_counts_single /= np.max(p_counts_single)
    he_counts_single /= np.max(he_counts_single)
    fe_counts_single /= np.max(fe_counts_single)
    
    p_counts_ensemble /= np.max(p_counts_ensemble)
    he_counts_ensemble /= np.max(he_counts_ensemble)
    fe_counts_ensemble /= np.max(fe_counts_ensemble)

    # Calculate cumulative histograms
    p_cumulative_single = np.cumsum(p_counts_single[::-1])[::-1]
    he_cumulative_single = np.cumsum(he_counts_single[::-1])[::-1]
    fe_cumulative_single = np.cumsum(fe_counts_single[::-1])[::-1]
    
    p_cumulative_ensemble = np.cumsum(p_counts_ensemble[::-1])[::-1]
    he_cumulative_ensemble = np.cumsum(he_counts_ensemble[::-1])[::-1]
    fe_cumulative_ensemble = np.cumsum(fe_counts_ensemble[::-1])[::-1]

    # Normalize cumulative histograms
    p_cumulative_single /= np.max(p_cumulative_single)
    he_cumulative_single /= np.max(he_cumulative_single)
    fe_cumulative_single /= np.max(fe_cumulative_single)
    
    p_cumulative_ensemble /= np.max(p_cumulative_ensemble)
    he_cumulative_ensemble /= np.max(he_cumulative_ensemble)
    fe_cumulative_ensemble /= np.max(fe_cumulative_ensemble)

    # Generate cumulative histogram plot
    plt.plot(p_bins_single[:-1], p_cumulative_single, label='p (SDNN)', color='blue', linestyle='dotted',linewidth=1.5)
    plt.plot(he_bins_single[:-1], he_cumulative_single, label='He (SDNN)', color='red', linestyle='dotted',linewidth=1.5)
    plt.plot(fe_bins_single[:-1], fe_cumulative_single, label='Fe (SDNN)', color='green', linestyle='dotted',linewidth=1.5)
    
    plt.plot(p_bins_ensemble[:-1], p_cumulative_ensemble, label='p (EDNN)', color='blue', linestyle='solid',linewidth=1.5)
    plt.plot(he_bins_ensemble[:-1], he_cumulative_ensemble, label='He (EDNN)', color='red', linestyle='solid',linewidth=1.5)
    plt.plot(fe_bins_ensemble[:-1], fe_cumulative_ensemble, label='Fe (EDNN)', color='green', linestyle='solid',linewidth=1.5)
    
    # Set plot labels and legend
    plt.xlabel('Predicted Probability',fontsize=12)
    plt.ylabel('Selection Efficiency',fontsize=10)

    plt.legend(loc='lower left',fontsize=8)
    # plt.title('Cumulative Distributions', fontsize=14)
    
    if prob_cut is not None:
        plt.xlim(prob_cut, 1)

    # Set y-axis to log scale
    plt.yscale('log')

    # Add gridlines
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    # Move the ticks to the inside of the plot
    plt.tick_params(axis='both', direction='in')

    # Add ticks on the upper and right parts of the plots
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_tick_params(which='both', top=True, bottom=True, direction='in')
    plt.gca().yaxis.set_tick_params(which='both', left=True, right=True, direction='in')
    plt.figure(figsize=(8,6))

    # Show the plot
    plt.tight_layout()
    plt.show()



def plot_evaluation_plots(model, X, y, figsize=(12, 10)):
    """ TODO: improve and complete this function"""
    
    # Make predictions using the trained model
    predictions = model.predict(X)  # X is your input data

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y.ravel(), predictions.ravel())

    # Compute F1 score
    f1 = f1_score(y.argmax(axis=1), predictions.argmax(axis=1), average='weighted')

    # Compute class-wise metrics
    class_labels = ['p', 'He', 'Fe']
    class_metrics = classification_report(y.argmax(axis=1), predictions.argmax(axis=1), target_names=class_labels, output_dict=True)

    # Compute calibration curve
    prob_positives = predictions.max(axis=1)
    true_labels = y.argmax(axis=1)

    # Compute feature importance
    #feature_importance = model.feature_importances_  # Replace with the appropriate method to obtain feature importances

    # Plotting the evaluation plots
    plt.figure(figsize=figsize)

    # Precision-Recall Curve
    plt.subplot(3, 2, 1)
    plt.plot(recall, precision, marker='.', color= 'green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # F1 Score
    plt.subplot(3, 2, 2)
    plt.bar(0, f1)
    plt.xticks([])
    plt.ylabel('F1 Score')
    plt.title('F1 Score')

    # Class-wise Metrics
    plt.subplot(3, 2, 3)
    plt.imshow(pd.DataFrame(class_metrics).iloc[:-1, :].values, cmap='Blues', vmin=0, vmax=1)
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
    plt.yticks(ticks=np.arange(len(class_metrics) - 1), labels=list(class_metrics[:-1]))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Class-wise Metrics')

    # Calibration Curve
    plt.subplot(3, 2, 4)
    fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, prob_positives, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, marker='.', linestyle='-', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
