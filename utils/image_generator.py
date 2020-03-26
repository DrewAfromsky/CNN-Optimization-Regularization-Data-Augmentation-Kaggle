#!/usr/bin/env/ python

# This Python script contains the ImageGenrator class.

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):
    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        self.x = x.copy()
        self.y = y.copy()
        self.num_samples, self.height, self.width, self.num_channels = self.x.shape
        self.trans_width = 0
        self.trans_height = 0
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
    
    def create_aug_data(self):
                
        '''
        Combine all the data to form a augmented dataset 
        '''
        
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        return self.x_aug, self.y_aug
                    
    def next_batch_gen(self, batch_size, shuffle=True):
        
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        total_batches = self.num_samples // batch_size
        batch_count = 0
        
        while True:
            if(batch_count < total_batches):
                batch_count += 1
                yield (self.x[(batch_count - 1) * batch_size : batch_count * batch_size],
                       self.y[(batch_count - 1) * batch_size : batch_count * batch_size])
            else:
                if shuffle:
                    p = np.random.permutation(self.num_samples)
                    x1 = x[perm,:,:,:]
                    y1 = y[perm]
                batch_count = 0
        return x1

    def show(self, images):
        
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """

        xshow = images[:16]
        fig = plt.figure(figsize=(10,10))

        for i in range(16):
            ax = fig.add_subplot(4,4,i+1)
            ax.imshow(xshow[i,:].reshape(28,28), 'gray')
            ax.axis('off')

    def translate(self, shift_height, shift_width):
        
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """
                
        self.trans_height += shift_height
        self.trans_width += shift_width
        translated = np.roll(self.x.copy(), (shift_width, shift_height), axis=(1, 2))
        print('Current translation: ', self.trans_height, self.trans_width)
        self.shift_height = shift_height
        self.shift_width = shift_width
        self.translated = (translated,self.y.copy())
        return translated

    def rotate(self, angle=0.0):
        
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        """
        
        self.dor = angle
        rotated = rotate(self.x.copy(), angle,reshape=False,axes=(1, 2))
        print('Currrent rotation: ', self.dor)
        self.rotated = (rotated, self.y.copy())
        return rotated

    def flip(self, mode='h'):
        
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        
        self.is_horizontal_flip = 'h' in mode
        self.is_vertical_flip = 'v' in mode
        if self.is_horizontal_flip:
            flipped = np.flip(self.x.copy(), axis = 2)
        if self.is_vertical_flip:
            flipped = np.flip(self.x.copy(), axis = 1)
        self.flipped = (flipped, self.y.copy())
        return flipped
    
    def add_noise(self, portion, amplitude):
        
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        
        s = np.int(np.ceil(self.num_samples * portion))
        p = np.random.permutation(s)
        added = self.x[p,:,:,:]
        ys = self.y[p,]
        mean = 0
        var = 0.1
        std = var ** 0.5
        gauss = np.random.normal(mean, std, (self.height, self.width, self.num_channels))
        gauss = gauss.reshape(self.height, self.width, self.num_channels)
        for k in range(added.shape[0]):
            added[k,:,:,:] = added[k,:,:,:] + amplitude * gauss
        self.added = (added, self.y.copy())
        return added
