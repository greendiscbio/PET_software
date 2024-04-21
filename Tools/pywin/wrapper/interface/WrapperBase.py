# Module that defines the basic interface that wrappers must implement.
#
# Author: Fernando García <ga.gu.fernando@gmail.com> 
#
from abc import ABCMeta, abstractmethod


class WrapperBase:
    """
    Base class for wrapper classes.
    """
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def algorithms(self):
        """
        Return all fitted algorithms.
        """
        pass

