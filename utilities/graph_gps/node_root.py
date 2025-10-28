from adios_io import read_adios_dataset, write_adios_data
import mpi_utils
import traceback
from logger import logger
from mpi4py import MPI

from utilities.graph_gps.adios_io import read_extra_attrs


def _create_task_list(pyg_chunk, task_size):
    """
    Split chunk/list of pyg objects into sublists of size task_size.
    Manage memory so that objects are removed from the original list and moved into the new sublists
    """
    task_list = []

    l = []
    while len(pyg_chunk) > 0:
        l.append(pyg_chunk.pop())

        if len(l) >= task_size:
            task_list.append(l)
            l = []

    if len(l) > 0:
        task_list.append(l)

    return task_list


def _get_next_worker():
    """
    Get the next worker that is ready for a task
    """
    status = MPI.Status()
    _ = mpi_utils.node_comm.recv(source=MPI.ANY_SOURCE, status=status)
    return status.Get_source()


def _assign_to_workers(label, pyg_chunk):
    """
    Split the chunk of pyg objects into a lists of objects and assign them to workers
    """
    task_size = min(len(pyg_chunk), len(pyg_chunk) // (2*mpi_utils.node_size))
    logger.debug(f"Node root on {mpi_utils.hostname} received {len(pyg_chunk)} {label} objects from adios. "
                 f"Sending tasks to workers in chunks of {task_size} objects")

    task_lists = _create_task_list(pyg_chunk, task_size)
    for task_list in task_lists:
        w = _get_next_worker()
        mpi_utils.node_comm.send((label, task_list), dest=w)
        logger.debug(f"Node root on {mpi_utils.hostname} assigned {len(task_list)} objects to worker {w}")


def _wait_for_workers_to_finish():
    """
    Wait for all workers to finish transforms and then signal them to quit
    """
    logger.debug(f"Node root on {mpi_utils.hostname} waiting for workers to finish")
    for _ in range(mpi_utils.node_size-1):
        w = _get_next_worker()
        mpi_utils.node_comm.send(None, dest=w)


def node_root(adios_in):
    """
    Root process of each node. It reads in pyg objects in chunks from the adios file and assigns them dynamically to
    worker processes on the node. When all objects have been transformed, the node root writes them into the output
    adios file.
    """
    try:
        for label in ('trainset', 'valset', 'testset'):
            for pyg_chunk in read_adios_dataset(adios_in, label):
                _assign_to_workers(label, pyg_chunk)

        _wait_for_workers_to_finish()
        logger.info(f"Graph transforms on {mpi_utils.hostname} done")

    except Exception as e:
        logger.error(f"Exception {e} at {traceback.format_exc()}")
        MPI.COMM_WORLD.Abort(1)
