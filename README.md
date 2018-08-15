# Behaviorial Cloning

The goals / steps of this project are the following:
* Use a driving simulator to collect data of good driving behavior 
* Train and validate a convolutional neural network that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the track in the simulator




[//]: # (Image References)

[original]: ./image/original_data_histogram.png "Histogram of original data"
[final]: ./image/after_data_augmentation.png "After Data Augmentation"
[center]: ./image/center.jpg "Example Image from Car Center Camera"

---

### Model Architecture

#### 1. Choosing an architecture

End-to-End Behavioral Cloning has been attempted by several research teams before, and trying to implement an architecture on my own would be naive. Perhaps the most famous model was made by a team at NVIDIA. Their [results](https://arxiv.org/pdf/1604.07316.pdf) show great performance from their model, but there is one problem: it is too large. I needed to train my model on a CPU, but given the size of their model, it would have taken way too long. I found a couple other similar models, and finally settled on [this one](https://github.com/commaai/research) made by comma.ai. It is significantly smaller than the NVIDIA model, and appeared to be just as effective.

The model begins with a normalization layer implemented as a Keras Lambda layer. I added a Cropping2D layer after the normalization layer in an effort to remove irrelevant data and further decrease training time. The model continues as a convolution neural network with 3 convolutional layers with 8x8, 5x5, and 5x5 kernel sizes, respectively. These layers are followed by only two fully connected layers, the first with 512 neurons, and the second is the output layer, with just a single output neuron for the predicted steering angle. All activation layers use the exponential linear unit (ELU) activation function. A 20% dropout rate is added after the final convolutional layer and a 50% dropout rate is added after the first fully connected layer. Finally, the loss function used is mean squared error (MSE).

The full Keras definition is found in model.py and repeated here:

```python
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding="same"))
model.add(ELU())
model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding="same"))
model.add(ELU())
model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
```


#### 2. Attempts to reduce overfitting in the model

As discussed previously, the model contains two dropout layers in order to reduce overfitting. 

The data was split 80/20 for train/validation to ensure that the model was not overfitting. Training output showed a marginally higher mean squared error on the validation set, which is to be expected. Both the training and validation loss decreased throughout training, for 7 epochs. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. A large majority of the data I used was just the default data that came with the simulator. After a few runs, I added some training data of me driving around only the sharp turns 3-4 times. I only used the center camera images from the default data and my own training data. An example image from data I took on a sharp turn from the center camera:

![alt_text][center]

For details about how I refined the training data, see the next section. 

---

### Model Training Strategy

#### 1. Solution Design Approach

My first step was to the model described above and train it on the default data, without capturing anymore data. The first run went surprisingly well, but the car eventually drove off the raod because it did not turn sharp enough. It appeared to have a bias toward driving straight. So I took a histogram of the data to see what values I was working with, and got this result:

![alt_text][original]

Seeing this histogram made it quite obvious why the car had a tendency to drive straight. At this point, I collected more data only on the sharp turns. I did this several times, re-training each time. This gradually increased the training time with only a marginal improvement each time. At this point, instead of deciding to add more data, I decided to throw out data I didn't want and augment the data that I did. I added a quick check in my function that loads images: if the absolute value of the steering measurement was less than 0.05, the assocaited image was given a 50% chance of being thrown out. If it was greater than 0.05, it was given an 80% to be copied into the training set again, but left-right flipped and with the steering angle multiplied by -1. This technique allowed me to increase the instances of turns in my data while decreasing the instances of straight driving. After gathering the data again, the distribution looked much more realistic:

![alt_text][final]

After this, the model performed very well! A video of a full lap can be found [here](./run.mp4). It still struggled on the challenge track a bit; the straight bias was still a little too high for some of the sharp turns and it could not complete a full lap. Further, I took no training data from the challenge track, so the different textures may have come into play. Regardless, I found it very impressive that the model could perform so well on only ~8000 test images. Further refinement of the data, and possibly adding more data, certainly some from the challenge track, would be necessary to complete a full lap on the challenge track.


#### 2. Final Model Architecture
The same as what I started with. To achieve better performance, I only modified the data.

#### 3. Result
See a recorded video [here](./run.mp4).

