import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, classification_report
from sklearn.calibration import calibration_curve
import seaborn as sns

def free_mem():
    return gc.collect()

def convert_data_to_npz(folder_path, file_names, output_file):
    """
    Converts the data files to a single .npz file or loads the existing .npz file.
    Returns the proton, helium, and iron arrays.
    
    Args:
        folder_path (str): The folder path where the data files are located.
        file_names (list): The list of file names to convert.
        output_file (str): The output .npz file name.
    """
    if not os.path.exists(os.path.join(folder_path, output_file)):
        data = {}
        
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            
            # Read the text file
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Read the rows
                rows = []
                for line in lines:
                    row_data = np.fromstring(line, sep=' ', dtype=np.float32)
                    rows.append(row_data)

                # Convert the rows list to a numpy array
                array_data = np.array(rows)

                # Store the array in the data dictionary
                data[file_name.split(".")[0]] = array_data
                print(f"Conversion completed successfully for file: {file_name}\n")

        # Save all the arrays as a single npz file
        np.savez(os.path.join(folder_path, output_file), **data)

        print(f"\nConversion completed successfully!\nFile saved in {output_file}")

    data = np.load(os.path.join(folder_path, output_file))

    p  = data['pr_events']
    He = data['he_events']
    Fe = data['fe_events']

    print(f"The iron file has a shape of {Fe.shape}")
    print(f"The helium file has a shape of {He.shape}")
    print(f"The proton file has a shape of {p.shape}")
    
    return p, He, Fe


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
    plt.xlabel('Background efficiency (FPR)', fontsize=12)
    plt.ylabel('Signal efficiency (TPR)',fontsize=10)
    #plt.title('Receiver Operating Characteristic (ROC)',fontsize=14)
    plt.legend()
    
    # Add AUC scores to the plot
    class_labels = ['p', 'He', 'Fe']
    for i, auc in enumerate(auc_scores):
        plt.text(0.5, 0.37-i*0.05, f'AUC class {class_labels[i]} = {auc*100:.2f} %', 
                horizontalalignment='left', verticalalignment='center')
    # Add ticks on the upper and right parts of the plots
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_tick_params(which='both', top=True, bottom=True, direction='in')
    plt.gca().yaxis.set_tick_params(which='both', left=True, right=True, direction='in')
    plt.figure(figsize=(8,6))
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
    plt.xlabel('Background efficiency (FPR)',fontsize=12)
    plt.ylabel('Signal efficiency (TPR)',fontsize=10)
    # plt.title('Receiver Operating Characteristic (ROC)',fontsize=14)
    plt.legend()

    # Add AUC scores to the plot
    class_labels = ['p', 'He', 'Fe']
    
    for i, auc_ensemble in enumerate(auc_scores_ensemble):
        plt.text(0.5, 0.37-i*0.05, f'EDNN AUC {class_labels[i]} = {auc_ensemble*100:.2f} %', 
                horizontalalignment='left', verticalalignment='bottom')
    
    # Add ticks on the upper and right parts of the plots
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_tick_params(which='both', top=True, bottom=True, direction='in')
    plt.gca().yaxis.set_tick_params(which='both', left=True, right=True, direction='in')
    plt.figure(figsize=(8,6))
    # Show the plot
    plt.tight_layout()
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