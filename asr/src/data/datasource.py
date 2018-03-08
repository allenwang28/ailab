# -*- coding: utf-8 *-* 

import abc

class DataSource(object):
    __metaclass__ = abc.ABCMeta
    self._input_shape = None
    self.output_shape = None

    @abstractmethod
    def batch_generator(self):
        raise NotImplementedError('implement the batch generator')
