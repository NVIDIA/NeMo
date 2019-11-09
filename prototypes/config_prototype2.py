# Problem: we want a general method that will enable creation of objects with
#   - pre-trained weights (done already - trainable params save/load in PT)
#   - configuration understood as set of attributes with their values (TODO)
#
# We want a mechanism that will go up the class hierarchy and save all
# configuration parameters that are provided by the intermediate classes.


import inspect
import numpy


class NeuralModule(object):

    # Constructor
    def __init__(self, name):
        self.name = name
        # By default, the Module is placed on CPU...
        self.device = "CPU"
        # ... and is GPU compatible, i.e. can be moved to GPU.
        self.gpu_compatible = True

    # To get name
    def get_name(self):
        return self.name

    # To check if as GPU compatible
    def is_gpu_compatible(self):
        return self.gpu_compatible

    def _configuration_iterator(self):
        """
        Iterates over configuration and yields the key and value pair.
        """
        # Iterate over the object properties, relying on the instrospection
        # mechanism.
        for key, val in inspect.getmembers(self, lambda x: not inspect.ismethod(x)):
            # Skip all "pythonic" objects.
            if not key.startswith('__'):
                yield key, val

    def print_configuration(self):
        """
        Prints the configuration.
        """
        print("Configuration for {} ({}):".format(self.name, id(self)))
        for key, val in self._configuration_iterator():
            print("{}: {}".format(key, val))
        print("\n")

    def export_configuration(self):
        """
        Method exports the configuration understood as a list of object
        attributes with their values.
        """
        return {key: val for key, val in self._configuration_iterator()}

    def import_configuration(self, config):
        """
        Imports configuration from the provided config.
        """
        for key, _ in self._configuration_iterator():
            if key in config.keys():
                # print(key, " current value: ",
                #      self.__getattribute__(key))
                #print(key, " setting: ", config[key])
                # Set the value.
                self.__setattr__(key, config[key])

    @classmethod
    def from_configuration(self, config):
        """
        Imports configuration from the provided config.
        """
        for key, _ in self._configuration_iterator():
            if key in config.keys():
                # print(key, " current value: ",
                #      self.__getattribute__(key))
                #print(key, " setting: ", config[key])
                # Set the value.
                self.__setattr__(key, config[key])


class TrainableNM(NeuralModule):

    # Constructor: one parameter is provided with default, the other one without.
    def __init__(self, name, attrib1="one", attrib2):
        # Call base class constructor
        super().__init__(name)

        # Add new attribute.
        self.attrib1 = attrib1
        self.attrib2 = attrib2
        self.attrib3 = "hardcoded attribute value 3"

        # Create the NN.


class NeuralFC(TrainableNM):

    # Constructor
    def __init__(self, name, attrib_b, input_dim=10, hidden_dim=15, output_dim=20):
        # Call base class constructor
        super().__init__(name)

        # Add new attribute.
        self.attrib_1 = "hardcoded value Module 2"
        self.attrib_b = attrib_b

        # Change default attr of base class - set that it is not compatible.
        self.gpu_compatible = False


# Assumption - all the "important" parameters from the point of view of operation
# of a given NM are set in the _init_,
# i.e. other class attributes, that have changed during the life of the object
# from _init_ up till now are not important/will not be saved...
