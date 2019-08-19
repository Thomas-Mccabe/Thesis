 #! /usr/bin/python

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Assigning the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def main():

    print(tf.__version__)

    # Import the mnist data
    fashion_mnist = keras.datasets.fashion_mnist

    # Classify the images and labels
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  
    print('The size of the dataset', train_images.shape)

    print("Number of labels: ",len(train_labels)
    )

    # Preprocessing the data 
    '''
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    '''

    # The aboive image shows that the data ranges 
    # From 0 to 255

    # Converting it to 0-1 for the NN model
    train_images = train_images / 255
    test_images = test_images/ 255

    # Plotting the first 25 to vairfy that they 
    # are what we expect
    '''
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    '''
    
    # Defining the model 
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

        # This is compiling the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

     # This will train the model
     model.fit(train_images, train_labels, epochs=5)

    # Now lets evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test Accuracy', test_acc)          

    predictions = model.predict(test_images)
    print(np.argmax(predictions[0])
    )

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    '''
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(i, predictions, test_labels)
        plt.show()
    '''

    # Grab an image from the test dataset
    img = test_images[0]
    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img,0))

    print(img.shape)

    predictions_single = model.predict(img)

    print(predictions_single)
    plot_value_array(0, predictions_single, test_labels)
    plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    prediction_result = np.argmax(predictions_single[0])
    print(prediction_result)
        
if __name__ == '__main__':
    main()