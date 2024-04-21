# Module that contains the implementation of the populations used in the algorithms
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## Module dependencies
from . import Population


class Block:
    """
    A block defines a set of features that are grouped together and can be treated as a single feature in
    the population.
    """
    def __init__(self, structure: dict, features: list):
        """
            __init__(structure, features)

        Parameters
        -------------
        :param structure : dict
            Dictionary where the keys define the name of the block and the values correspond to the features that
            will be treated as a block.
        :param features: 1d-array
            Name of the columns of the original data. All must be present in the values of structure param.
        """
        self.block = {}
        for idx, feat in enumerate(features):
            # Get the block to which a certain feature belongs
            key, idx_feat = Block._get_key(structure, feat)

            # Annotate feature inside a block maintaining its original position in the data.
            if self.block.get(key, None) is None:
                self.block[key] = [(idx, structure[key][idx_feat])]
            else:
                self.block[key].append((idx, structure[key][idx_feat]))

    def __repr__(self):
        return f"Block(num_of_blocks:{len(self.block.keys())})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """
        Comparision between two blocks.

        Parameters
        ------------
        :param other: object

        Returns
        --------
        :return bool
        """
        if isinstance(other, __class__):
            # Blocks content comparision
            if other.block == self.block:
                return True

        return False

    @property
    def blocks(self):
        """
        Return the blocks defined as keys in structure.

        Returns
        ---------
        :return list
            Block names (keys from structure)
        """
        return list(self.block.keys())

    def index(self, features: list):
        """
        Returns the indices where the features of the selected block are found in the original data.

        Parameters
        -------------
        :param features: 1d-array
            List of features (name of blocks = keys from structure).

        Returns
        ---------
        :return list
            Original position of the elements that form a block in the data.
        """
        return self._get_from_block(n=0, features=features)

    def values(self, features: list):
        """
        Returns the values (column names) where the features of the selected blocks are found in the original data.

        Parameters
        -------------
        :param features: 1d-array
            List of features (name of blocks = keys from structure).

        Returns
        ---------
        :return list
            Block feature names in the original data
        """
        return self._get_from_block(n=1, features=features)

    def _get_from_block(self, n, features):
        """
        Returns the names of the features that make up a block (n=0) or their index (m=1).

        Parameters
        ------------
        :param n: int
            n=1 to get the name or n=0 to get the indices.
        :param features: 1d-array

        Returns
        ----------
        :return elements: list
            Block members names or indices.
        """
        return [value[n] for feat in features for value in self.block[feat]]

    @staticmethod
    def _get_key(dic: dict, target_val: str):
        """
        Returns the name of the block to which a feature defined in structure belongs. If it cannot find the
        name, that is, not all the features defined in structure are present, it will throw an error.

        Parameters
        -----------
        :param dic: dict
            User-defined structure passed to the constructor
        :param target_val: str
            Target value

        Returns
        ---------
        :return str
            Block name.
        """
        target_key, target_idx = None, None
        for key, value in dic.items():
            for idx, feat in enumerate(value):
                # A feature is found inside a block
                if feat == target_val:
                    # If a feature is duplicated it will raise an error
                    if target_key is not None:
                        raise Exception("Duplicated value: %s and %s values "
                                        "must be unique." % (feat, target_key))
                    # Annotate feature and feature position
                    target_key, target_idx = key, idx

        # If feature hasn't been found
        if target_key is None:
            raise Exception("A feature from col_feats is not present in dictionary (%s). "
                            "Please check that all values are present" % target_val)

        return target_key, target_idx

    @classmethod
    def merge(cls, block_1, block_2):
        """
        Merge two block.

        Parameters
        ------------
        :param block_1: Block
        :param block_2: Block

        Returns
        --------
        :return Block
        """
        new_block = {**block_1.block, **block_2.block}

        return new_block


class BlockPopulation(Population):
    """
    Class that uses a Block as features (see Block documentation).
    """
    def __init__(self, size: int, min_feat_init: int = 2, max_feat_init: int = None,
                 structure: dict = None, features: list = None, block: Block = None):
        """
        __init__(size: int, min_feat_init: int = 2, max_feat_init: int = None, structure: dict = None,
                    features: list = None, block: Block = None)

        Notes
        ------
        This population can be initialized from an instance of Block or directly from a structure (dictionary)
        and a list of features.
        """
        # Select maximum number of features in initialization
        if max_feat_init is None:
            max_feat_init = len(structure.keys())

        # Initialize population
        super().__init__(size=size, min_feat_init=min_feat_init, max_feat_init=max_feat_init)

        # Select features using a Block or a structure + list of features
        if not (structure is None and features is None):
            self._features = Block(structure=structure, features=features)
        else:
            self._features = block

    @property
    def features(self):
        """
        Return the features of an entire block

        Returns
        --------
        :return dict
        """
        return {n: block for n, block in enumerate(self._features.blocks)}

    @features.setter
    def features(self, new_features):
        """
        Select new features using a dictionary or a Block.

        Parameters
        -----------
        :param new_features: dict or Block
        """

        # Update max_feat_init (for security)
        if self.max_feat_init is None:
            self.max_feat_init = len(new_features)

        # If new features come as a dictionary, transform it in a Block
        # (implemented for GlobalizationAlg compatibility)
        if isinstance(new_features, dict):
            self._features.block = new_features
        else:
            # If the new features come as a Block assign directly to features
            self._features = new_features

    @property
    def features_num(self):
        """
        Return features numerical indices.

        Returns
        ---------
        :return list
        """
        return [n for n in range(len(self._features.block))]

    @property
    def block(self):
        """
        Return a Block.

        Returns
        ---------
        :return Block
        """
        return self._features

    def init_features(self, features: list):
        """
        Method not implemented for this class.
        """
        pass

    def generate_random_population(self, size: int):
        """
        This method generates a new random population with a given size.

        Parameters
        -----------
        :param size: int

        Returns
        --------
        :return BlockPopulation
        """
        if size == 0:
            return None

        # Create new population
        random_population = BlockPopulation(size=size, block=self._features,
                                            min_feat_init=self.min_feat_init,
                                            max_feat_init=self.max_feat_init)

        # Initializes the population individuals
        random_population.init(include_all=False)

        return random_population

    def create_new_population(self, size: int, individuals: list):
        """
        Method that creates a new population using a given size and individuals.

        Parameters
        -----------
        :param size: int
        :param individuals tuple
            ([features], fitness)

        Returns
        --------
        :return BlockPopulation
        """
        # Create population
        new_population = BlockPopulation(size=size, block=self._features,
                                         max_feat_init=self.max_feat_init,
                                         min_feat_init=self.min_feat_init)
        # Set individuals
        new_population.set_new_individuals(individuals)

        return new_population

    @staticmethod
    def merge_population(population_1, population_2):
        """
    Method that takes two populations and mixes them.

        Parameters
        -----------
        :param population_1: BlockPopulation
        :param population_2: BlockPopulation

        Returns
        ---------
        :return BlockPopulation
        """
        # Merge individuals and their fitness
        individuals = population_1.individuals + population_2.individuals

        # Sum population sizes
        size = population_1.size + population_2.size

        # Merge population features
        if not population_1.block == population_2.block:
            new_block = Block.merge(population_1.block, population_2.block)
        else:
            new_block = population_1.block

        # Create a new population
        mixed_population = BlockPopulation(size=size, block=new_block,
                                           min_feat_init=population_1.min_feat_init,
                                           max_feat_init=population_1.max_feat_init)

        # Assign individuals and their fitness
        mixed_population.set_new_individuals(individuals)

        return mixed_population

