from functools import partial
from typing import List


class CallbackManager:
    """
    Callbacks Manager transparently operating on non-wired callbacks.
    Contract: Callbacks are processed as a filter Chain::
        (ArgA, ArgB) -> callbackA -> callbackB -> ... -> (ArgA', ArgB')
    methods must return updated values in the same order as in method signature.

    """

    def __init__(self):
        self.callbacks: List[any] = []

    def register_callback(self, callback: object, index: int = None):
        """
        Registers callback for operaton. Callback later can be called::

            argA, argB = callback.method(argA, argB)

        if there are no callbacks registered, all function calls will act transparently and return same values, passed into the function.

        Parameters
        ----------
        callback: obj
            callback instance implementing callback methods.
        index: int (optional)
            position what position to add the callback - use this to inject high-priority preprocessing.
            Defaults to None - this will add new callback at the end of the filter chain.
        """
        if index is None:
            self.callbacks.append(callback)
        else:
            self.callbacks.insert(index, callback)

    def __getattr__(self, operation):
        if operation.startswith('__') or operation in ['register_callback']:
            return super().item
        else:
            return partial(self.__internal_callbacks_call, operation)

    def __internal_callbacks_call(self, operation, *args):
        for c in self.callbacks:
            if hasattr(c, operation):
                call = getattr(c, operation)
                args = call(*args)
        if type(args) == list:
            if len(args) == 1:
                args = args[0]
        return args
