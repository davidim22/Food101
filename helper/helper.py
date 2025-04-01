import itertools
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Function to import and resize the image for our model
def import_and_resize_image(filepath, img_shape=224, scale=True):
    """
    Reads the image from the filepath. Turns the image into tensor
    and resizes the image int (img_shape, img_shape, 3).

    Parameters
    ----------
    filepath (str): string filepath of the target image
    img_shape (int): size to resize the target image (default: 224)
    scale (bool): scale the pixels in the range of 0 to 1 (default: true)

    Returns
    ----------
    an image from filepath that is resized and scaled (default=True)
    """
    # Read the image and decode it to tensorf
    img = tf.io.read_file(filepath)
    img = tf.image.resize(img, [img_shape, img_shape])

    # Rescale the image
    if scale:
        return img/255.
    else:
        return img

# Function to create confusion matrix
def create_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=15, norm=False, savefig=False):
    """
    Creates a labelled confusion matrix that will compare the predictions
    and ground truth labels.

    Parameters
    ----------
    y_true (array): truth labels (need to have the same shape as y_pred)
    y_pred (array): predicted labels (need to have the same shape as y_true)
    classes (array): class labels (default: set to None) else integer labels
    figsize (tuple): size of the figure output (default=(10x10))
    text_size (int): size of the figure text (default=15)
    norm (bool): normalize the values (default=False) or not
    savefig (bool): saves the confusion matrix (default=False) or not

    Returns
    ----------
    a labelled confusion matrix comparing y_true and y_pred
    """
    # Create the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # Normalize --> divide the matrix by a vector [[nx1]] where n in the len of the matrix
    cf_matrix_norm = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # Number of classes in the confusion matrix
    num_classes = cf_matrix.shape[0]

    # Create a plot
    fig, ax = plt.subplot(figsize=figsize)
    cax = ax.matshow(cf_matrix, cmap=plt.cf_matrix.Blue) # The color represents how correct it is
    fig.colorbar(cax)

    # Dealing with classes
    if classes:
        labels = classes
    else:
        labels = np.arange(num_classes)

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="y_pred",
           ylabel="y_true",
           xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabel=labels,
           yticklabel=labels)
    
    # X-axis to the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Setting the threshold
    threshold = (cf_matrix.max() + cf_matrix.min()) / 2.

    # Plot the text 
    for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
        if norm:
            plt.text(j, i, f"{cf_matrix_norm[i. j]} ({cf_matrix_norm[i. j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cf_matrix[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cf_matrix[i, j]}",
                     horizontalalignment="center",
                     color="white" if cf_matrix[i, j] > threshold else "black",
                     size=text_size)
    
    # Saving the figure
    if savefig:
        fig.savefig("confusion_matrix.png")

# Predict the image and plot them
def pred_and_plot(model, filepath, class_names):
    """
    Import the image from the filepath. Make prediction of the image on the
    trained model. Plot the image with the predicted class as the title.

    Parameters
    ----------
    model (model): the machine learning model
    filepath (str): the location of the image
    class_name (list): list of the class names
    """
    # Import the target image preprocess it
    img = import_and_resize_image(filepath)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    
    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

# Create tensorboard callback
def start_tensorboard_callback(dir_name, experiment_name):
    """
    Create a TensorBoard Callback instant to store log files. The log
    files are stored in the following as:
        "dir_name/experiment_name/current_datetime/"

    Parameter
    ---------
    dir_name (str): directory to store the TensorBoard log files
    experiment_name (str): name of the experiment directory

    Return
    ---------
    Returns the TensorBoard Callback
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving the TensorBoard log files into: {log_dir}")
    return tensorboard_callback

# Plot the validation and the training data separately
def plot_loss_curves(history):
    """
    Returns separate loss curve for the training and the validation metrics

    Parameter
    ---------
    history: TensorFlow model History object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot the loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot the accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

# Compare the history of two TensorFlow models
def compare_history(original_history, new_history, initial_epochs=5):
    """
    Compares the history of the two TensorFlow models

    Parameters
    ----------
    original_history: history object from original model (before new_history)
    new_history: history object from continued model training (after original_history)
    initial_epochs: number of epochs in original_history
    """

    # Get the original history measurements
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']
    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']

    # Combine original and new history
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    # Plot the total history
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label='Training Accuarcy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plt([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Function to unzip a zipfile into the current directory
def unzip_data(filepath):
    """
    Unzips the files in the filepath into the current directory

    Parameter
    ---------
    filepath (str): location of the file to unzip
    """
    zip_ref = zipfile.ZipFile(filepath, "r")
    zip_ref.extractall()
    zip_ref.close()

# View the amount of files in a directiory and subdirectory
def dir_image_count(dir_path):
    """
    Goes through dir_path and returns the content within the dir

    Parameter
    ---------
    dir_path (str): target directory

    Return
    ---------
    Prints the following information
        number of subdirectory in the dir_path
        number of images/files in each subdirectory
        name of the subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}.")

# Function to evaluate: accuarcy, precision, recall, and f1-score
def calculate_results(y_true, y_pred):
    """
    Calculates the model accuracy, precision, recall, and f1-score

    Parameter
    ---------
    y_true: true labels (1d)
    y_pred: predicted labels (1d)

    Return
    ---------
    Returns a dictionary of accuracy, precision, recall, and f1-score
    """
    # Calculate the model accuracy
    model_accuracy = accuracy_score(y_true, y_pred)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weight')
    model_result = {
        'accuracy': model_accuracy,
        'precision': model_precision,
        'recall': model_recall,
        'model_f1': model_f1
    }
    return model_result