#!/usr/bin/env/ python

# This script contains forward/backward functions of regularization techniques

import numpy as np


def dropout_forward(x, dropout_config, mode):

    """
    Dropout feedforward
    :param x: input tensor with shape (N, D)
    :param dropout_config: (dict)
                           enabled: (bool) indicate whether dropout is used.
                           keep_prob: (float) retention rate, usually range from 0.5 to 1.
    :param mode: (string) "train" or "test"
    :return:
    - out: a tensor with the same shape as x
    - cache: (train phase) cache a random dropout mask used in feedforward process
             (test phase) None
    """
    
    keep_prob = dropout_config.get("keep_prob", 0.7)

    out, cache = None, None
    if mode == "train":

        cache = np.random.binomial(1,keep_prob,x.shape)/keep_prob
        out = x * cache

    elif mode == "test":

        out = x

    return out, cache


def dropout_backward(dout, cache):
    
    """
    Dropout backward only for train phase.
    :param dout: a tensor with shape (N, D)
    :param cache: (tensor) mask, a tensor with the same shape as x
    :return: dx: the gradients transfering to the previous layer
    """

    dx = cache * dout

    return dx


def bn_forward(x, gamma, beta, bn_params, mode):
    
    """
    Batch Normalization forward
    
    moving_mean = decay*moving_mean + (1-decay)*current_mean
    moving_var = decay*moving_var + (1-decay)*current_var
           
    :param x: a tensor with shape (N, D)
    :param gamma: (tensor) a scale tensor of length D, a trainable parameter in batch normalization.
    :param beta:  (tensor) an offset tensor of length D, a trainable parameter in batch normalization.
    :param bn_params:  (dict) including epsilon, decay, moving_mean, moving_var.
    :param mode:  (string) "train" or "test".
    
    :return:
    - out: a tensor with the same shape as input x.
    - cahce: (tuple) contains (x, gamma, beta, eps, mean, var)
    """
    
    eps = bn_params.get("epsilon", 1e-5)
    decay = bn_params.get("decay", 0.9)

    N, D = x.shape
    moving_mean = bn_params.get('moving_mean', np.zeros(D, dtype = x.dtype))
    moving_var = bn_params.get('moving_var', np.ones(D, dtype = x.dtype))

    out, mean, var = None, None, None
    if mode == "train":
        
        #############################################################
        #       Batch normalization forward train mode              #
        #      1. calculate mean and variance of input x            #
        #      2. normalize x with mean and variance                #
        #      3. apply scale(gamma) and offset(beta) on the        #
        #         normalized data                                   #
        #      4. use moving average method to update               #
        #         moving_mean and moving_var in the bn_params       #
        #############################################################
        
        current_mean=np.mean(x,axis=0)
        current_var=np.var(x,axis=0)
        out = gamma*(x-current_mean) / np.sqrt(current_var+eps) + beta
        moving_mean = decay * moving_mean + (1-decay) * current_mean
        moving_var = decay * moving_var + (1-decay) * current_var

    elif mode == 'test':
        
        #############################################################
        #       Batch normalization forward test mode               #
        #############################################################
        
        out = gamma * (x-moving_mean) / np.sqrt(moving_var+eps) + beta

    # Cache for back-propagation
    cache = (x, gamma, beta, eps, mean, var)
    
    # Update mean and variance estimation in bn_config
    bn_params['moving_mean'] = moving_mean
    bn_params['moving_var'] = moving_var

    return out, cache

def bn_backward(dout, cache):
    
    """
    Batch normalization backward
    Derive the gradients wrt gamma, beta and x

    :param dout:  a tensor with shape (N, D)
    :param cache:  (tuple) contains (x, gamma, beta, eps, mean, var)
    
    :return:
    - dx, dgamma, dbeta
    """
    
    x, gamma, beta, eps, mean, var = cache
    
    N, D = dout.shape
    dx, dgamma, dbeta = None, None, None

    x_hat = (x - mean) / np.sqrt(np.tile(var, (N, 1)) + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dout * gamma / np.sqrt(np.tile(var, (N, 1)) + eps)

    return dx, dgamma, dbeta
