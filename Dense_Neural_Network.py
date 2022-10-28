import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset, dataset included in keras
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # we have 10 Classes, so 10 Output neuron


# we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1. We can do this by dividing each value in the training and testing sets by 255.0. We do this because smaller values will make it easier for the model to process our values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
model = keras.Sequential([  # sequential means the data goes from the left to the right side.
   # Input Layer
    keras.layers.Flatten(input_shape=(28, 28)),  # The flatten means that our layer will reshape the shape (28,28) array into a vector of 784 neurons so that each pixel will be associated with one neuron.
   # Hidden Layer
    keras.layers.Dense(128, activation='relu'),  # The dense denotes that this layer will be fully connected and each neuron from the previous layer connects to each neuron of this layer
   # Output Layer
    keras.layers.Dense(10, activation='softmax') # This is our output later and is also a dense layer. It has 10 neurons that we will look at to determine our models output. Each neuron represnts the probabillity of a given image being one of the 10 different classes. The activation function softmax is used on this layer to calculate a probabillity distribution for each class. This means the value of any neuron in this layer will be between 0 and 1, where 1 represents a high probabillity of the image being that class.
])
# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
# model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) # The verbose argument is defined from the keras documentation as: "verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."
print('Test accuracy:', test_acc) # Training accuracy is higher, this is because of overfitting. Overfitting happens because the training model sees the data 10 times/10 epoch, to fix this I could lower the amount of epochs/change parameters

# Making predictions
predictions = model.predict(test_images)  # model.predict([test_images[0]]), to predict for singular image we have to place in an array


COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
