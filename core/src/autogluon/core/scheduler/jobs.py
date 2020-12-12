import logging
import multiprocessing as mp
import os
import random
import sys
import time
from functools import partial
from time import sleep

import dill

from autogluon.core import make_temp_directory, Task
from autogluon.core.scheduler.managers import TaskManagers
from autogluon.core.scheduler.reporter import Communicator, LocalStatusReporter
from autogluon.core.utils import CustomProcess, AutoGluonEarlyStop
from autogluon.core.utils.multiprocessing_utils import is_fork_enabled
import threading

logger = logging.getLogger(__name__)

__all__ = ['DistributedJobRunner']


class DistributedJobRunner(object):

    JOB_REGISTRATION_LOCK = mp.Lock()

    def __init__(self, task: Task, managers: TaskManagers):
        self.task: Task = task
        self.managers: TaskManagers = managers

    def start_distributed_job(self):
        """Async Execute the job in remote and release the resources
        """
        logger.debug(f'Scheduling {self.task}')

        def _release_resource_callback(fut):
            logger.debug(f'Releasing Resources {self.task.task_id}')
            with DistributedJobRunner.JOB_REGISTRATION_LOCK:
                self.managers.release_resources(self.task.resources)

        with DistributedJobRunner.JOB_REGISTRATION_LOCK:
            remote: Remote = self.task.resources.node
            job = remote.submit(
                partial(self._run_dist_job, self.task.task_id),
                self.task.fn,
                self.task.args,
                self.task.resources.gpu_ids
            )
            # Fast job submission from multiple threads causes lost jobs and hangs reporter thread.
            # Keep the values above 200ms combined
            time.sleep(0.15)
            job.add_done_callback(_release_resource_callback)
            time.sleep(0.15 + random.random())

        return job

    @staticmethod
    def _run_dist_job(task_id, fn, args, gpu_ids):
        """Remote function Executing the task
        """
        try:
            manager = mp.Manager()
            return_list = manager.list()

            if '_default_config' in args['args']:
                args['args'].pop('_default_config')

            if 'reporter' in args:
                local_reporter = LocalStatusReporter()
                dist_reporter = args['reporter']
                args['reporter'] = local_reporter

            # Starting local process
            # Note: we have to use dill here because every argument passed to a child process over spawn or forkserver
            # has to be pickled. fork mode does not require this because memory sharing, but it is unusable for CUDA
            # applications (CUDA does not support fork) and multithreading issues (hanged threads).
            # Usage of decorators makes standard pickling unusable (https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)
            # Dill enables sending of decorated classes. Please note if some classes are used in the training function,
            # those classes are best be defined inside the function - this way those can be constructed 'on-the-other-side'
            # after deserialization.
            pickled_fn = fn if is_fork_enabled() else dill.dumps(fn)

            # Reporter has to be separated since it's used for cross-process communication and has to be passed as-is
            args_ = {k: v for (k, v) in args.items() if k not in ['reporter']}
            pickled_args = args_ if is_fork_enabled() else dill.dumps(args_)

            cross_process_args = {k: v for (k, v) in args.items() if k not in ['fn', 'args']}

            with make_temp_directory() as tempdir:
                p = CustomProcess(
                    target=partial(DistributedJobRunner._worker, tempdir, task_id, pickled_fn, pickled_args),
                    args=(return_list, gpu_ids, cross_process_args)
                )
                p.start()
                if 'reporter' in args:
                    cp = Communicator.Create(p, local_reporter, dist_reporter)

                if p.is_alive():
                    p.join()

                if 'reporter' in args:
                    cp.stop()
                    if cp.is_alive():
                        cp.join()

                # Get processes outputs
                if not is_fork_enabled():
                    DistributedJobRunner.__print(tempdir, task_id, 'out')
                    DistributedJobRunner.__print(tempdir, task_id, 'err')
        except Exception as e:
            logger.error('Exception in worker process: {}'.format(e))
        ret = return_list[0] if len(return_list) > 0 else None
        return ret

    @staticmethod
    def __print(tempdir, task_id, out):
        with open(os.path.join(tempdir, f'{task_id}.{out}')) as f:
            out = f.read()
            file = sys.stderr if out is 'err' else sys.stdout
            if out:
                print(f'(task:{task_id})\t{out}', file=file, end='')

    @staticmethod
    def _worker(tempdir, task_id, pickled_fn, pickled_args, return_list, gpu_ids, args):
        """Worker function in the client
        """
        with open(os.path.join(tempdir, f'{task_id}.out'), 'w') as std_out:
            with open(os.path.join(tempdir, f'{task_id}.err'), 'w') as err_out:
                if not is_fork_enabled():
                    sys.stdout = std_out
                    sys.stderr = err_out

                # Only fork mode allows passing non-picklable objects
                fn = pickled_fn if is_fork_enabled() else dill.loads(pickled_fn)
                args = {**pickled_args, **args} if is_fork_enabled() else {**dill.loads(pickled_args), **args}

                DistributedJobRunner.set_cuda_environment(gpu_ids)

                # running
                try:
                    ret = fn(**args)
                except AutoGluonEarlyStop:
                    ret = None
                return_list.append(ret)

                sys.stdout.flush()
                sys.stderr.flush()

    @staticmethod
    def set_cuda_environment(gpu_ids):
        if len(gpu_ids) > 0:
            # handle GPU devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"