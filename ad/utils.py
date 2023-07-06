import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, classification_report
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
import seaborn as sns

def free_mem():
    return gc.collect()

def reload_models(num_models, model_folder_path):
    models = []

    for i in range(num_models):
        model_path = os.path.join(model_folder_path, f"model_{i+1}_latest")
        model_architecture_path = os.path.join(model_path, 'model_architecture.json')
        model_weights_path = os.path.join(model_path, 'model_weights.h5')

        # Load model architecture from JSON file
        with open(model_architecture_path, 'r') as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json)

        # Load model weights
        model.load_weights(model_weights_path)

        models.append(model)

    return models

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
    plt.xlabel('QM1[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M1' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    if cut != None:
        plt.axvline(x=cut, color='r', linestyle='--',label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('QM2[Phe]')
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
    free_mem()

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
    plt.xlabel('QM1[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M1' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    plt.axvline(x=cut, color='r', linestyle='--',label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('QM2[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M2' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    plt.axvline(x=cut, color='b', linestyle='--',label='axvline2 - full height')

    plt.tight_layout()
    plt.show()

    return filtered_data

def doWeights(model, mini = None, maxi = None, log = False, figsize=(6, 3)):
    """Function for plotting the weight distributions"""
    allWeightsByLayer = {}
    i = 0
    for layer in model.network.layers:
        if (layer._name).find("batch") != -1 or len(layer.get_weights()) < 1:
            continue 
        weights = layer.weights[0].numpy().flatten()  
        allWeightsByLayer[layer._name] = weights
        i += 1
    
    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])
    
    if mini == None:
        mini = np.min(np.concatenate(histosW)) 
    print(f'Minimum weight value: {mini}')
    if maxi == None:
        maxi = np.max(np.concatenate(histosW)) 
    print(f'Maximum weight value: {maxi}')

    fig = plt.figure(figsize=figsize)
    
    bins = np.linspace(mini, maxi, 150)
    histosW = np.array(histosW, dtype='object')

    colors = plt.get_cmap('turbo')(np.linspace(0.5, 0.9, len(histosW)))  # Update this line

    for i in range(len(histosW)):
        plt.hist(histosW[i], bins, histtype='stepfilled', stacked=True, label=labelsW[i], edgecolor='black', color=colors[i])
    
    plt.legend(frameon=False, loc='upper right', fontsize='small')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    if log:
        plt.semilogy()
    
    plt.figtext(0.4, 0.58, model._name, wrap=True, horizontalalignment='left', verticalalignment='center', fontsize='medium')
    #plt.grid(True)

    plt.show()

def WhiskerWeights(model, figsize=(6, 3)):
    """Function for plotting the Whisker plot of weights"""
    allWeightsByLayer = {}
    i=0
    for layer in model.network.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[0].numpy().flatten()  
        allWeightsByLayer[layer._name] = weights
        i+=1
    labelsW = []
    dataW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        dataW.append(allWeightsByLayer[key])
    
    plt.figure(figsize=figsize)
    
    # Create a whisker plot using the data
    dataW = np.array(dataW, dtype='object')
    bplot = plt.boxplot(dataW, labels=labelsW, vert=False, meanline=True, patch_artist=True)
    plt.xlabel('Weights')
    plt.ylabel('Layers')
    plt.figtext(0.4, 0.55, model._name, wrap=True, horizontalalignment='left',verticalalignment='center')

    # fill with colors
    colors = plt.get_cmap('turbo')(np.linspace(0.5, .9, i))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.show()

def plot_metrics(model, metric='accuracy', figsize=(2, 2)):
    """Function for plotting the trend of different metrics"""
    if metric not in model.history:
        print(f"Metric '{metric}' not found in model history.")
        return

    train_metric = model.history[metric]
    val_metric = model.history[f'val_{metric}']
    epochs = range(1, len(train_metric) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, train_metric, 'bo-', label=f'Train {metric.capitalize()}')
    plt.plot(epochs, val_metric, 'ro-', label=f'Valid {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    #plt.grid(True)
    plt.show()

def visualize_class_distribution( y, predictions, figsize=(3, 4), true_color='blue', predicted_color='red'):
    # Get the predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)  # Assuming one-hot encoded labels

    # Calculate the class counts for true labels
    true_class_counts = np.bincount(np.argmax(y, axis=1))  # Assuming one-hot encoded labels
    true_class_labels = ['p', 'He', 'Fe']

    # Calculate the class counts for predicted labels
    predicted_class_counts = np.bincount(predicted_labels)
    predicted_class_labels = true_class_labels

    # Plot the distribution of true class labels
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.bar(true_class_labels, true_class_counts, color=true_color)
    plt.xlabel('True Class')
    plt.ylabel('Count')
    plt.title('True Class Distribution')
    plt.yscale('log')

    # Plot the distribution of predicted class labels
    plt.subplot(1, 2, 2)
    plt.bar(predicted_class_labels, predicted_class_counts, color=predicted_color)
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.title('Predicted Class Distribution')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


def plot_auc( y, predictions, figsize=(5, 5)):
    # Calculate the AUC score for each class
    num_classes = y.shape[1]
    auc_scores = []
    for i in range(num_classes):
        auc = roc_auc_score(y[:, i], predictions[:, i])
        auc_scores.append(auc)
    
    # Compute false positive rate and true positive rate for ROC curve
    fpr, tpr, _ = roc_curve(y.ravel(), predictions.ravel())
    
    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUC = {np.mean(auc_scores)*100:.2f} %')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.plot([0, 0, 1], [0, 1, 1], 'r--', label='Perfect Classifier')
    plt.xlabel('Background efficiency (FPR)')
    plt.ylabel('Signal efficiency (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    
    # Add AUC scores to the plot
    class_labels = ['p', 'He', 'Fe']
    for i, auc in enumerate(auc_scores):
        plt.text(0.5, 0.5-i*0.05, f'AUC class {class_labels[i]} = {auc*100:.2f} %', 
                horizontalalignment='left', verticalalignment='center')
    
    plt.show()

def plot_auc_compared(y, predictions_single, predictions_ensemble, figsize=(5,5)):
    # Calculate the AUC score for each class
    num_classes = y.shape[1]
    auc_scores_single = []
    auc_scores_ensemble = []
    
    for i in range(num_classes):
        auc_single = roc_auc_score(y[:, i], predictions_single[:, i])
        auc_ensemble = roc_auc_score(y[:, i], predictions_ensemble[:, i])
        
        auc_scores_single.append(auc_single)
        auc_scores_ensemble.append(auc_ensemble)
    
    # Compute false positive rate and true positive rate for ROC curve
    fpr_single, tpr_single, _ = roc_curve(y.ravel(), predictions_single.ravel())
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y.ravel(), predictions_ensemble.ravel())
    
    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr_single, tpr_single, label=f'Single DNN AUC = {np.mean(auc_scores_single)*100:.2f} %')
    plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble DNN AUC = {np.mean(auc_scores_ensemble)*100:.2f} %')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.plot([0, 0, 1], [0, 1, 1], 'r--', label='Perfect Classifier')
    plt.xlabel('Background efficiency (FPR)')
    plt.ylabel('Signal efficiency (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()

    # Add AUC scores to the plot
    class_labels = ['p', 'He', 'Fe']
    
    for i, auc_ensemble in enumerate(auc_scores_ensemble):
        plt.text(0.5, 0.55-i*0.05, f'EDNN AUC {class_labels[i]} = {auc_ensemble*100:.2f} %', 
                horizontalalignment='left', verticalalignment='center')
    
    plt.show()



def plot_confusion_matrix(y, predictions, num_classes=3, normalize=False, figsize=(5, 3)):
    # Compute and plot the confusion matrix
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', cbar=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    class_labels = ['p', 'He', 'Fe']
    plt.xticks(np.arange(num_classes) + 0.5, class_labels)
    plt.yticks(np.arange(num_classes) + 0.5, class_labels)

    plt.tight_layout()
    plt.show()



def plot_evaluation_plots(model, X, y, figsize=(12, 10)):
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

def plot_rejection_power(y, predictions_single, predictions_ensemble, prob_cut=None, figsize=(5, 5)):
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
    plt.plot(p_bins_single[:-1], p_cumulative_single, label='p (SDNN)', color='blue', linestyle='dotted')
    plt.plot(he_bins_single[:-1], he_cumulative_single, label='He (SDNN)', color='red', linestyle='dotted')
    plt.plot(fe_bins_single[:-1], fe_cumulative_single, label='Fe (SDNN)', color='green', linestyle='dotted')
    
    plt.plot(p_bins_ensemble[:-1], p_cumulative_ensemble, label='p (EDNN)', color='blue', linestyle='solid')
    plt.plot(he_bins_ensemble[:-1], he_cumulative_ensemble, label='He (EDNN)', color='red', linestyle='solid')
    plt.plot(fe_bins_ensemble[:-1], fe_cumulative_ensemble, label='Fe (EDNN)', color='green', linestyle='solid')
    
    # Set plot labels and legend
    plt.xlabel('Predicted Probability')
    plt.ylabel('Rejection Power')

    plt.legend(loc='lower left')
    
    if prob_cut is not None:
        plt.xlim(prob_cut, 1)

    # Set y-axis to log scale
    plt.yscale('log')

    # Show the plot
    plt.show()



def plot_normalized_histograms(predictions_single, predictions_ensemble, prob_cut=None):
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

    plt.title('Score Histogram', fontsize=14)

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
    plt.figure(figsize=(8,6))

    # Show the plot
    plt.tight_layout()
    plt.show()