#!/usr/bin/env/ python

# This Python script contains various functions for layer construction.

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

#### Code was completed by Drew Afromsky for the assignment for Nerual Networks and Deep Learning, ECBM 4040, @Columbia University, Fall 2019 ###

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Here, 'pad' means the number rows/columns of zeroes to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    To calculate the output shape of convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    """

    batch, height, width, channels = x.shape
    filt_height, filt_width, channels, num_filt = w.shape

    new_height = (height-filt_height + 2 * pad) // stride + 1
    new_width = (width - filt_width + 2 * pad) // stride + 1

    new_x = np.zeros((batch, new_height, new_width, num_filt))
    
    x_pad = np.zeros((batch, height + 2 * pad, width+2 * pad, channels))
    
    for bt in range(batch):
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    x_pad[bt, i+pad, j+pad, c] = x[bt, i, j, c]
                    
    for bt in range(batch):
        for ft in range(num_filt):
            for i in range(new_height):
                for j in range(new_width):
                    new_x[bt, i, j, ft] = b[ft] + np.sum(w[:,:,:,ft] * 
                                   x_pad[bt, i*stride: i*stride + filt_height, j*stride: j*stride + filt_width,:])
    
    return new_x

def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    """

    batch, height, width, channels = x.shape
    
    filter_height, filter_width, channels_f,num_of_filters = w.shape
    batch_dtop, new_height, new_width, num_f_top = d_top.shape
    
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    x_pad = np.zeros((batch, height + 2*pad, width+2*pad, channels))
    for bt in range(batch):
        for i in range(height):
            for j in range(width):
                for cn in range(channels):
                    x_pad[bt,i+pad,j+pad,cn] = x[bt,i,j,cn]
                    
    for bt in range(batch):
        for ft in range(num_of_filters):
            for i in range(new_height):
                for j in range(new_width):
                    dw[:,:,:,ft] += d_top[bt,i,j,ft] * x_pad[bt,i*stride :i*stride + filter_height,j * stride: j*stride + filter_width,:]
                    db[ft] += d_top[bt,i,j,ft]
    
    return dw, db, dw.shape 

def max_pool_forward(x, pool_size, stride):
    """
    A Numpy implementation of 2-D image max pooling.

    Inputs:
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform max operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
    """

    batch, height, width, channels = x.shape

    new_height = (height-pool_size) // stride + 1
    new_width = (width-pool_size) // stride + 1
    
    pool = np.zeros((batch, new_height, new_width, channels))

    for l in range(batch):
        for k in range(channels):
            for i in range(new_height):
                i_tmp = i * stride
                for j in range(new_width):
                    j_tmp = j * stride
                    pool[l,i,j,k]=np.max(x[l, i_tmp:i_tmp+pool_size, j_tmp:j_tmp+pool_size, k])

    return pool