# -*- coding: utf-8 *-* 
"""Implementation of Baidu's Deep Speech 2 model

Todo:
    - Add support for more than just BasicRNNCell       
    - Add support for dropout parameters
    - Add support for hyperparameter options
    - Create a new class for parameters/parsing to take care of all of the above...
"""
import tensorflow as tf

from models.tf_decorate import define_scope
from models.tf_class import TensorflowModel

class DeepSpeech2(TensorflowModel):
    def _construct(self):
        # conv1
        with tf.variable_scope('deepspeech_conv1') as scope:
            l1_filter = tf.get_variable('deepspeech_l1_filter', shape=(42, 11, 1, 32))
            l1_stride = [1,2,2,1]
            conv1 = tf.nn.conv2D(self.input_tensor, l1_filter, l1_stride, padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1, training=self._training)
            conv1 = tf.contrib.layers.dropout(conv1, keep_prob=0.8, is_training=self._training)
        # conv2
        with tf.variable_scope('deepspeech_conv2') as scope:
            l2_filter = tf.get_variable('deepspeech_l2_filter', shape=(21, 11, 32, 32))
            l2_stride = [1,2,1,1]
            conv2 = tf.nn.conv2D(self.input_tensor, l2_filter, l2_stride, padding='SAME')
            conv2 = tf.layers.batch_normalization(conv2, training=self._training)
            conv2 = tf.contrib.layers.dropout(conv2, keep_prob=0.8, is_training=self._training)
        # conv3
        with tf.variable_scope('deepspeech_conv3') as scope:
            l3_filter = tf.get_variable('deepspeech_l3_filter', shape=(21, 11, 32, 96))
            l3_stride = [1,2,1,1]
            conv3 = tf.nn.conv2D(self.input_tensor, l3_filter, l3_stride, padding='SAME')
            conv3 = tf.layers.batch_normalization(conv3, training=self._training)
            conv3 = tf.contrib.layers.dropout(conv3, keep_prob=0.8, is_training=self._training)
        # 4 recurrent layers
        # recurrent1
        with tf.variable_scope('deepspeech_recurrent1') as scope:
            r1_cell = tf.contrib.nn.BasicRNNCell(256)
            recurrent1 = tf.nn.dynamic_rnn(r1_cell, conv3, sequence_length=self._max_output_length, time_major=True)
            recurrent1 = tf.layers.batch_normalization(recurrent1, training=self._training)
            recurrent1 = tf.contrib.layers.dropout(recurrent1, keep_prob=0.8, is_training=self._training)
        # recurrent2
        with tf.variable_scope('deepspeech_recurrent2') as scope:
            r2_cell = tf.contrib.nn.BasicRNNCell(256)
            recurrent2 = tf.nn.dynamic_rnn(r2_cell, recurrent1, sequence_length=self._max_output_length, time_major=True)
            recurrent2 = tf.layers.batch_normalization(recurrent2, training=self._training)
            recurrent2 = tf.contrib.layers.dropout(recurrent2, keep_prob=0.8, is_training=self._training)
        # recurrent3
        with tf.variable_scope('deepspeech_recurrent3') as scope:
            r3_cell = tf.contrib.nn.BasicRNNCell(256)
            recurrent3 = tf.nn.dynamic_rnn(r3_cell, recurrent2, sequence_length=self._max_output_length, time_major=True)
            recurrent3 = tf.layers.batch_normalization(recurrent3, training=self._training)
            recurrent3 = tf.contrib.layers.dropout(recurrent3, keep_prob=0.8, is_training=self._training)
        # recurrent4
        with tf.variable_scope('deepspeech_recurrent4') as scope:
            r4_cell = tf.contrib.nn.BasicRNNCell(256)
            recurrent4 = tf.nn.dynamic_rnn(r4_cell, recurrent3, sequence_length=self._max_output_length, time_major=True)
            recurrent4 = tf.layers.batch_normalization(recurrent4, training=self._training)
            recurrent4 = tf.contrib.layers.dropout(recurrent4, keep_prob=0.8, is_training=self._training)
        # fully connected
        with tf.variable_scope('deepspeech_fc') as scope:
            fully_connected = tf.layers.dense(recurrent4, self._num_output_features)
        return fully_connected

    def _predictions(self):
        return tf.to_int32(tf.nn.ctc_beam_search_decoder(self._output_node, self._max_output_length, merge_repeated=False)[0][0])

    def _optimize(self):
        """
        if gradient_clip == -1: # Don't apply gradient clipping
            return tf.train.AdamOptimizer(learning_rate).minimize(self._loss)
        else:
        """
        gradient_clip = 1
        learning_rate = 0.001
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self._loss, self._var_trainable_op), gradient_clip)
        return tf.train.AdamOptimizer(learning_rate)

    def _loss(self):
        return tf.reduce_mean(tf.nn.ctc_loss(self._target_tensor, self._output_node, self.max_output_length))
