import torch
import time
from pytorch_lightning.callbacks.callback import Callback
from nemo.lightning import io



def dump_thread_stacks():
    for th in threading.enumerate():
       print(th)
       traceback.print_stack(sys._current_frames()[th.ident])

def hangman_main(interval_ms):
    assert interval_ms > 0
    while True:
       time.sleep(interval_ms / 1000)
       dump_thread_stacks()

class Hangman(Callback, io.IOMixin):
    """
    Hangman checks thread stacks at regular inteval (specified via interval_ms) to detect whether a function
    is "hanging" at some point of code. Its main purpose is to aid debugging.

    Args:
        interval_ms (float): Interval to check for hanging in ms.

    Example:
        >>> callback = Hangman(250)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, interval_ms: float = 250):
        """
        interval (int): How frequently to check DDP weights for errors. Default to 0 (off).
        """
        assert interval_ms > 0, "Expected interval to be > 0. A zero interval makes DdpParityChecker a no-op."

        self.thread = Thread(target=hangman_main, args = (interval_ms, ))
        self.thread.start()
