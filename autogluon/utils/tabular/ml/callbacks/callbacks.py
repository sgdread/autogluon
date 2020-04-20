from functools import partial
from typing import Dict


class CallbackList:

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __getattr__(self, item):
        if item.startswith('__'):
            return super().item
        else:
            return partial(self.__internal_callbacks_call, item)

    def __internal_callbacks_call(self, operation, *args):
        for c in self.callbacks:
            if hasattr(c, operation):
                call = getattr(c, operation)
                # print(args)
                args = call(*args)
        return args


class CallbackManager:
    """
    Callbacks Manager transparently operating on non-wired callbacks.
    Contract: Callbacks are processed as a filter Chain::
        (ArgA, ArgB) -> callbackA -> callbackB -> ... -> (ArgA', ArgB')
    methods must return updated values in the same order as in method signature.

    """

    def __init__(self):
        self.callbacks: Dict[str, list] = {}

    def register_callback(self, namespace: str, callback: object, index: int = None):
        """
        Registers callback for operaton. Callback later can be called::

            argA, argB = callback.namespace.method(argA, argB)

        if there are no callbacks registered, all function calls will act transparently and return same values, passed into the function.

        Parameters
        ----------
        namespace: str
            operations namespace
        callback: obj
            callback instance implementing callback methods.
        index: int (optional)
            position what position to add the callback - use this to inject high-priority preprocessing.
            Defaults to None - this will add new callback at the end of the filter chain.
        """
        op_callbacks: list = self.callbacks.get(namespace, list())
        if index is None:
            op_callbacks.append(callback)
        else:
            op_callbacks.insert(index, callback)
        self.callbacks[namespace] = op_callbacks

    def __getattr__(self, operation):
        if operation.startswith('__'):
            return super().item
        else:
            return CallbackList(self.callbacks.get(operation, []))

    def __internal_callbacks_call(self, operation, *args, **kwargs):
        return CallbackList(self.callbacks.get(operation, []))
