import threading

from nemo.core.neural_factory import DeviceType


class Singleton(type):
    """ Implementation of a generic singleton meta-class. """

    # List of instances - one per class.
    __instances = {}
    # Lock used for accessing the instance.
    __lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """ Returns singleton instance.A thread safe implementation. """
        if cls not in cls.__instances:
            # Enter critical section.
            with cls.__lock:
                # Check once again.
                if cls not in cls.__instances:
                    # Create a new object instance - one per class.
                    cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # Return the instance.
        return cls.__instances[cls]


class AppState(metaclass=Singleton):
    """
        Application state stores variables important from the point of view of execution of the NeMo application.
        Staring from the most elementary (epoch number, episode number, device used etc.) to the currently
        active graph etc.
    """

    def __init__(self, device=DeviceType.GPU):
        """
        Constructor. Initializes global variables.
        """
        self.device = device
        self.active_graph = None
