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
        plt.axvline(x=cut, color='r', label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('QM2[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M2' or 'Histogram of M2')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    
    if cut != None:
        plt.axvline(x=cut, color='b', label='axvline2 - full height')
    

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
    plt.axvline(x=cut, color='r', label='axvline1 - full height')

    # Plot M2 histogram
    plt.subplot(1, 2, 2)
    plt.hist(M2, bins=bins, color='red', range=range, alpha=0.7, density=hnormalize, histtype='step')
    plt.xlabel('QM2[Phe]')
    plt.ylabel('Entries')
    plt.title(title + ' histogram of M2' or 'Histogram of M1')  # Set the title provided or use a default value
    if ylog:
        plt.yscale('log')
    plt.axvline(x=cut, color='b', label='axvline2 - full height')

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

    colors = plt.get_cmap('turbo')(np.linspace(0.1, 0.9, len(histosW)))  # Update this line

    for i in range(len(histosW)):
        plt.hist(histosW[i], bins, histtype='stepfilled', stacked=True, label=labelsW[i], edgecolor='black', color=colors[i])
    
    plt.legend(frameon=False, loc='upper right', fontsize='small')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    if log:
        plt.semilogy()
    
    plt.figtext(0.4, 0.38, model._name, wrap=True, horizontalalignment='left', verticalalignment='center', fontsize='medium')
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
    
    fig = plt.figure(figsize=figsize)
    
    # Create a whisker plot using the data
    dataW = np.array(dataW, dtype='object')
    bplot = plt.boxplot(dataW, labels=labelsW, vert=False, meanline=True, patch_artist=True)
    plt.xlabel('Weights')
    plt.ylabel('Layers')
    plt.figtext(0.4, 0.25, model._name, wrap=True, horizontalalignment='left',verticalalignment='center')

    # fill with colors
    colors = plt.get_cmap('turbo')(np.linspace(0.1, .9, i))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.show()

def plot_metrics(model, metric='accuracy', figsize=(3, 3)):
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

def visualize_class_distribution(model, X, y, figsize=(6, 3), true_color='blue', predicted_color='red'):
    # Make predictions using the trained model
    predictions = model.predict(X)  # X is your input data

    # Get the predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)  # Assuming one-hot encoded labels

    # Calculate the class counts for true labels
    true_class_counts = np.bincount(np.argmax(y, axis=1))  # Assuming one-hot encoded labels
    true_class_labels = ['p', 'He', 'Fe']

    # Calculate the class counts for predicted labels
    predicted_class_counts = np.bincount(predicted_labels)
    predicted_class_labels = ['p', 'He', 'Fe']

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

def plot_auc_with_confusion_matrix(model, X, y, figsize=(12, 5)):
    # Make predictions using the model
    predictions = model.predict(X)
    
    # Calculate the AUC score for each class
    num_classes = y.shape[1]
    auc_scores = []
    for i in range(num_classes):
        auc = roc_auc_score(y[:, i], predictions[:, i])
        auc_scores.append(auc)
    
    # Compute false positive rate and true positive rate for ROC curve
    fpr, tpr, _ = roc_curve(y.ravel(), predictions.ravel())
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ROC curve
    ax1.plot(fpr, tpr, label=f'AUC = {np.mean(auc_scores)*100:.2f} %')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend()
    
    # Add AUC scores to the plot
    class_labels = ['p', 'He', 'Fe']
    for i, auc in enumerate(auc_scores):
        ax1.text(0.5, 0.3-i*0.05, f'AUC class {class_labels[i]} = {auc*100:.2f} %', 
                horizontalalignment='left', verticalalignment='center')

    # Compute and plot the confusion matrix
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    ax2.set_title('Confusion Matrix')
    ax2.set_xticks(np.arange(num_classes) + 0.5)
    ax2.set_yticks(np.arange(num_classes) + 0.5)
    ax2.set_xticklabels(class_labels)
    ax2.set_yticklabels(class_labels)
    
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
