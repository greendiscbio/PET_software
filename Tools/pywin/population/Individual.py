# Module that contains the implementation of the populations used in the algorithms
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## Module dependencies


class Individual:
    """
    DESCRIPTION
    """
    def __init__(self, features):
        """
        DESCRIPTION
        :param features:
        """
        self.features = features
        self._fitness = None

    def __repr__(self):
        return f"Individual(features={self.features} fitness={self._fitness}"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        """
        DESCRIPTION
        :param item:
        :return:
        """
        return self.features[item]

    def __len__(self):
        """
        DESCRIPTION
        :return:
        """
        return len(self.features)

    @property
    def fitness(self):
        """
        DESCRIPTION
        :return:
        """
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        """
        DESCRIPTION
        :param new_fitness:
        :return:
        """
        self._fitness = new_fitness

