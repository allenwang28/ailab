# -*- coding: utf-8 *-* 
# -*- coding: utf-8 *-* 
"""Implementation of Baidu's Deep Speech model
"""

from tf_class import TensorflowModel

class DeepSpeech(TensorflowModel):

    @define_scope(initializer=tf.contrib.xavier_initializer())
    def predict(self):
        with tf.variable_scope('deepspeech_conv1') as scope:

    @define_scope()
    def optimize(self):


    @define_scope()
    def error(self):
