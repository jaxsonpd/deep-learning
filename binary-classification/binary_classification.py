## 
# @file binary_classification.py
# @author Jack Duignan (JackpDuignan@gmail.com)
# @date 2024-09-12
# @brief Use a signle layer neural network for binary classification

import numpy as np
import gzip
import matplotlib.pyplot as plt

class BinaryClassificationModel():
    def __init__(self) -> None:
        self.input_size = 28*28
        self.num_classes = 1
        self.batch_size = 100
        self.learning_rate = 0.005

        self.W = np.ones((self.num_classes, self.input_size))*0
        self.b = np.ones((self.num_classes, 1))*0
        

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform one iteration of the model

        ### Params
        inputs[nxinput_size]
            An array of inputs
        
        ### Returns
        probabilities [nxnum_classes]
            The result of the call
        """
        return inputs @ self.W.T + self.b.T
    
    def back_propergation(self, inputs: np.ndarray, labels: np.ndarray, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform back propergation over the data

        ### Params
        inputs [nxinput_size]
            The array of inputs
        labels [nx1]
            The labels for the inputs
        outputs [nxnum_classes]
            The result of the call operation

        ### Returns
        gradW
            The gradient of the weight vector
        gradb 
            The gradient of the bias vector
        """
        y = labels - (outputs > 1)

        gradW = y.T @ inputs
        gradb = y.T @ np.ones((len(inputs), 1))

        return gradW, gradb

    def gradient_descent(self, gradW: np.ndarray, gradb: np.ndarray) -> None:
        """
        Update the weights and biases

        ### Params:
        gradW
            The gradient of the weights
        gradb
            The gradient of the bias
        """
        self.W += self.learning_rate * gradW
        self.b += self.learning_rate * gradb

    def accuracy(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the accuracy of the model

        ### Params:
        outputs
            The output of the call function
        labels
            The labels to calculate the accuracy against

        ### Return
        accuracy
            The accuracy from 0 - 1 
        """
        results = np.equal(outputs > 1, labels)

        return np.sum(results)/len(labels)
        
def train(model: BinaryClassificationModel, train_inputs: np.ndarray, train_labels: np.ndarray, epochs: int):
    """
    Train the neural network

    ### Params:
    model
        The model to train
    train_inputs
        The inputs to train on
    train_labels
        The labels of the training data
    epochs
        The number of epochs to train for
    """

    for epoch in range(epochs):
        for index in range(0, len(train_inputs), model.batch_size):
            inputs = train_inputs[index:index+model.batch_size]
            labels = train_labels[index:index+model.batch_size]

            outputs = model.call(inputs)
            gradW, gradb = model.back_propergation(inputs, labels, outputs)
            print(gradW)
            model.gradient_descent(gradW, gradb)


def test(model, test_inputs, test_labels) -> float:
    """
    Test the model
    
    ### Params
    model
        The model under test
    test_inputs
        The inputs to use in the test
    test_labels
        The labels to use in the test

    ### Returns
    accuracy
        The accuracy for the model 0 to 1
    """

    outputs = model.call(test_inputs)
    return model.accuracy(outputs, test_labels)

def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    images_f = gzip.open(inputs_file_path, 'rb')
    labels_f = gzip.open(labels_file_path, 'rb')

    images_f.read(16)
    labels_f.read(8)

    images_buffer = images_f.read(784*num_examples)
    images = np.frombuffer(images_buffer, dtype=np.uint8).reshape([num_examples, 784])    
    images = images.astype('float32')

    labels_buffer = labels_f.read(num_examples)
    labels = np.frombuffer(labels_buffer, dtype=np.uint8)

    normalized_images = images / 255

    return normalized_images, labels

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main(training_data_folder: str):
    
    train_inputs, train_labels = get_data(training_data_folder+"/train-images-idx3-ubyte.gz", training_data_folder+"/train-labels-idx1-ubyte.gz", 60000)
    test_inputs, test_labels = get_data(training_data_folder+"/t10k-images-idx3-ubyte.gz", training_data_folder+"/t10k-labels-idx1-ubyte.gz", 10000)

    train_inputs = train_inputs.reshape(-1, 784)
    test_inputs = test_inputs.reshape(-1, 784)

    train_mask = np.logical_or(train_labels == 0, train_labels == 1)
    test_mask = np.logical_or(test_labels == 0, test_labels == 1)

    train_inputs = train_inputs[train_mask]
    train_labels = train_labels[train_mask]

    test_inputs = test_inputs[test_mask]
    test_labels = test_labels[test_mask]

    model = BinaryClassificationModel()

    train(model, train_inputs, train_labels, 25)

    accuracy = test(model, test_inputs, test_labels)

    print(f"Accuracy after {25} epochs {accuracy*100}%")

    results = model.call(test_inputs[:10])
    visualize_results(test_inputs[:10], results, test_labels[:10])

if __name__ == "__main__":
    main("../data/MNIST_data")
