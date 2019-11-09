

# Problem: we want a general method that will enable creation of objects with
#   - pre-trained weights (done already - trainable params save/load in PT)
#   - configuration understood as set of attributes with their values (TODO)
# We want a mechanism that will go up the class hierarchy and save all
# configuration parameters that are provided by the intermediate classes.

# Note: The proposed solution will work for simple objects now.
# If we want to store more complex, then maybe we want e.g. to pickle then?

# Note 2: I have assumed that all params are exported/imported - for now.
# If we decide that we can have

import inspect


class AbstractModel(object):

    # Constructor
    def __init__(self, name):
        self.name = name
        # By default, the model is placed on CPU...
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


class Model1(AbstractModel):

    # Constructor
    def __init__(self, name, attrib1="one"):
        # Call base class constructor
        super().__init__(name)

        # Add new attribute.
        self.attrib1 = attrib1
        self.attrib2 = "hardcoded value model 1"


class Model2CPUOnly(AbstractModel):

    # Constructor
    def __init__(self, name, attrib_b="two"):
        # Call base class constructor
        super().__init__(name)

        # Add new attribute.
        self.attrib_1 = "hardcoded value model 2"
        self.attrib_b = attrib_b

        # Change default attr of base class - set that it is not compatible.
        self.gpu_compatible = False


m1 = Model1("Geek1")  # An Object of Person
# print(m1.get_name(), " can run on GPU: ", m1.is_gpu_compatible())


# m2 = Model2CPUOnly("Geek2")  # An Object of Employee
# print(m2.get_name(), " can run on GPU: ", m2.is_gpu_compatible())

# Now do something with m1, e.g. "move it to GPU" and change the value of its attribute.
print("Initial configuration")
m1.print_configuration()

print("Changes in the configuration")
m1.device = "changed GPU"
m1.attrib1 = "changed neo_oen"
m1.print_configuration()

# "Save configuration to checkpoint".
m1_config_checkpoint = m1.export_configuration()
# print("Saved configuration: ", m1_config_checkpoint)


# Create new object and load it from config.
m3 = Model1("Geek3")


print("Before import")
m3.print_configuration()

print("After import")
m3.import_configuration(m1_config_checkpoint)
m3.print_configuration()
