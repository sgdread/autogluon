import os
import socket
import logging
import multiprocessing as mp

from .remote import Remote
from ...utils import warning_filter

__all__ = ['RemoteManager']

logger = logging.getLogger(__name__)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


class RemoteManager(object):
    NODES = {}
    LOCK = mp.Lock()
    PORT_ID = mp.Value('i', 8700)
    MASTER_IP = None
    __instance = None
    def __new__(cls):
        # Singleton
        with cls.LOCK:
            if cls.__instance is None:
                cls.__instance = object.__new__(cls)
        cls.MASTER_IP = get_ip()
        cls.start_local_node()
        return cls.__instance

    @classmethod
    def get_master_node(cls):
        return cls.NODES[cls.MASTER_IP]

    @classmethod
    def start_local_node(cls):
        port = cls.get_port_id()
        with warning_filter():
            remote = Remote(cls.MASTER_IP, port, local=True)
        print('start_local_node waiting lock')
        with cls.LOCK:
            print('start_local_node got lock')
            cls.NODES[cls.MASTER_IP] = remote
        print('start_local_node released lock')

    @classmethod
    def launch_each(cls, launch_fn, *args, **kwargs):
        for node in cls.NODES.values():
            node.submit(launch_fn, *args, **kwargs)

    @classmethod
    def get_remotes(cls):
        return list(cls.NODES.values())

    @classmethod
    def upload_files(cls, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        for node in cls.NODES.values():
            node.upload_files(files, **kwargs)

    @classmethod
    def add_remote_nodes(cls, ip_addrs):
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        remotes = []
        for node_ip in ip_addrs:
            if node_ip in cls.NODES.keys():
                logger.warning('Already added remote {}'.format(node_ip))
                continue
            port = cls.get_port_id()
            remote = Remote(node_ip, port)
            print('add_remote_nodes waiting lock')
            with cls.LOCK:
                print('add_remote_nodes got lock')
                cls.NODES[node_ip] = remote
            print('add_remote_nodes released lock')
            remotes.append(remote)
        return remotes
    
    @classmethod
    def shutdown(cls):
        for node in cls.NODES.values():
            node.shutdown()
        cls.NODES = {}
        cls.__instance = None

    @classmethod
    def get_port_id(cls):
        print('get_port_id waiting lock')
        with cls.LOCK:
            print('get_port_id got lock')
            cls.PORT_ID.value += 1
            port_id = cls.PORT_ID.value
        print('get_port_id released lock')
        return port_id

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for node in self.NODES.values():
            node.shutdown()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n'
        for node in self.NODES.values():
           reprstr += '{}, \n'.format(node)
        reprstr += ')\n'
        return reprstr
