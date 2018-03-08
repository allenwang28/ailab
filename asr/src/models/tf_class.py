# -*- coding: utf-8 *-* 
"""Abstract class for a tensorflow model
"""

import abc
import tensorflow as tf
from tf_decorate import define_scope
import tf_util

class TensorflowModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.predict
        self.optimize
        self.error

    @abc.abstractmethod
    def construct_model(self):
        raise NotImplementedError('construct the model here')

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_sample(self):
        pass

    @define_scope(initializer=tf.contrib.xavier_initializer())
    def build_graph(self):

    @abc.abstractmethod
    @define_scope()
    def optimize(self):
        raise NotImplementedError('specify the optimizer here')

    @abc.abstractmethod
    @define_scope()
    def error(self):
        raise NotImplementedError('specify error function here')
