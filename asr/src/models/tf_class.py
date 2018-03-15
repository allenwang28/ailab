# -*- coding: utf-8 *-* 
"""Abstract class for a tensorflow model
"""

import abc
from abc import ABC, abstractmethod

import tensorflow as tf
from models.tf_decorate import define_scope
import models.tf_util
import time
import os


class TensorflowModel(ABC):
    __metaclass__ = abc.ABCMeta

    @property
    def save_dir(self):
        return os.path.join(SAVE_DIR_BASE, self.name)

    def __init__(self, data_source, save_dir, training=True, reset=True):
        self._data_source = data_source
        self._max_output_length = data_source.max_output_length
        self._max_input_length = data_source.max_input_length
        self._batch_size = data_source.batch_size
        self._num_features = data_source.num_features
        self._num_output_features = data_source.num_output_features
        self._training = training
        self._reset = reset
        self._save_dir = save_dir


    def train(self, num_epochs):
        self.build_graph()
        batch_gen = self._data_source.batch_generator(tf=True, randomize=True)
        with tf.Session(graph=self._graph) as sess:
            if reset:
                print ('Initializing model')
                sess.run(self._initial_op)
            else:
                ckpt = tf.train.get_checkpoint_state(self._save_dir)

            for epoch in range(num_epochs):
                start = time.time()
                print ("Epoch: {0}".format(epoch))

                for batch in batch_gen:
                    inputs, outputs = batch

                    feed_dict = {self._input_tensor : inputs,
                                 self._output_tensor : outputs}
            
                    _, l, pred, y, err = sess.run([self._optimizer, ])

    def predict(self):
        pass

    def predict_sample(self):
        pass

    @define_scope()
    def build_graph(self):
        self._graph = tf.Graph()
        with self.graph.as_default():
            # Specify the input 
            self._input_tensor = tf.placeholder(tf.float32,
                                               shape=self._data_source.input_shape)

            # Construct the model
            self._output_node = self._construct()

            self._target_tensor = tf.placeholder(tf.float32,
                                                 shape=self._data_source.output_shape)

            self._var_trainable_op = tf.trainable_variables()

            # Specify loss function
            self._loss = self._loss()

            # Specify optimizer 
            self._optimizer = self._optimize()

            self._initial_op = tf.contrib.layers.xavier_initializer()

            self._predictions = tf.argmax(inputs=self._output_node, axis=1)

    @abstractmethod
    def _construct(self, tf_in):
        raise NotImplementedError('construct the model here')

    @abstractmethod
    def _optimize(self):
        raise NotImplementedError('specify the optimizer here')

    @abstractmethod
    def _loss(self):
        raise NotImplementedError('specify loss function here')

