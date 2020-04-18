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

    def __init__(self):
        self.callbacks: Dict[str, list] = {}

    def register_callback(self, operation, callback, index=None):
        op_callbacks: list = self.callbacks.get(operation, list())
        if index is None:
            op_callbacks.append(callback)
        else:
            op_callbacks.insert(index, callback)
        self.callbacks[operation] = op_callbacks

    def __getattr__(self, operation):
        if operation.startswith('__'):
            return super().item
        else:
            return CallbackList(self.callbacks.get(operation, []))

    def __internal_callbacks_call(self, operation, *args, **kwargs):
        return CallbackList(self.callbacks.get(operation, []))
