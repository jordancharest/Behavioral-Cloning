"""A neural network used for behavioral cloning"""
# for building the neural net
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, ELU, Cropping2D

# for data wrangling
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

import random


def gather_training_data(log_path='./data/'):
    
    # first gather the default data
    with open(log_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader]

    images = []
    measurements = []
    for line in lines[1:]:
        measurement = float(line[3])
        
        # remove 50% of the driving straight images
        if abs(measurement) <= 0.05 and random.random() < 0.50:
            continue
    
        image = cv2.imread('./data/' + line[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
       
        measurements.append(measurement)
        
        # sometimes add the images to the training data twice, but left-right flipped
        if abs(measurement) > 0.05 and random.random() < 0.75:
            image_flip = np.fliplr(image)
            images.append(image_flip)
            measurements.append(-measurement)
       

         
       
    # then my training data
    with open(log_path + 'TRAIN/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader]

    for line in lines[1:]:
        measurement = float(line[3])
        
        # remove 50% of the driving straight images
        if abs(measurement) <= 0.05 and random.random() < 0.50:
            continue
        
        image = cv2.imread(line[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
       
        measurements.append(measurement)
        
        # sometimes add the images to the training data twice, but left-right flipped
        if abs(measurement) > 0.05 and random.random() < 0.75:
            image_flip = np.fliplr(image)
            images.append(image_flip)
            measurements.append(-measurement)
    
    return images, measurements
  
def show_histogram(measurements):
    plt.hist(measurements, bins=29)
    plt.title('Histogram of Steering Measurements')
    plt.xlabel('Steering Measurement')
    plt.ylabel('# of occurences')
    plt.grid(color='k', linestyle='-', linewidth=0.5)
        


''' 
    ---------------------------------------------------------------------------
    MAIN
    ---------------------------------------------------------------------------
'''
if __name__ == "__main__":
    
    train = True
    
    # extract the data
#    images, measurements = gather_training_data()
    
    # Examine the data
#    show_histogram(measurements)
    
    # data marshalling
    X_train = np.array(images)
    y_train = np.array(measurements) 

    
  
    if train:
        ## Neural Network Definition ---------------------------------------------
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((75,25), (0,0))))
        model.add(Conv2D(16, 8, 8, stride=(4, 4), padding="same"))
        model.add(ELU())
        model.add(Conv2D(32, 5, 5, stride=(2, 2), padding="same"))
        model.add(ELU())
        model.add(Conv2D(64, 5, 5, stride=(2, 2), padding="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))
    
        # Compile and train
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
        
        
        model.save('model.h5')