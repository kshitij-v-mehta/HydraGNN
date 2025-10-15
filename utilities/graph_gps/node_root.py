from torch_geometric.testing import minPython

from adios_io import read_adios_data, write_adios_data
import mpi_utils
import traceback
from logger import logger
from mpi4py import MPI
import time


def node_root(adios_in, adios_out):
    try:
        # Read adios data
        task_size = 10000
        t1 = time.time()
        datasets_in = read_adios_data(adios_in)
        t2 = time.time()
        if mpi_utils.node_roots_rank == 0: logger.info(f"ADIOS reading done in {round(t2-t1)} seconds.")

        # Hold transformed pyg objects in a new dict
        datasets_out = dict()
        for k in datasets_in.keys():
            if k == 'extra_attrs':
                datasets_out[k] = datasets_in[k]
                continue
            datasets_out[k] = []

        # iterate over trainset, valset, and testset and send pyg objects to workers
        t1 = time.time()
        logger.debug(f"Node root on {mpi_utils.hostname} sending tasks to workers")
        status = MPI.Status()
        workers_free = 0
        for k in datasets_in.keys():
            if k == 'extra_attrs': continue

            # Split list of PyG objects and send to workers
            dataset = datasets_in[k]
            while len(dataset) > 0:
                # Receive ping or completed tasks from a worker
                data_t = mpi_utils.node_comm.recv(source=MPI.ANY_SOURCE, status=status)
                if data_t is not None:
                    key, transformed_pyg_obj = data_t
                    datasets_out[key].extend(transformed_pyg_obj)
                else:
                    pass  # this was a ready ping from a worker
                worker = status.Get_source()
                workers_free += 1

                # Send a chunk of objects to the worker
                pyg_chunks = []
                next_task_size = min(task_size, len(dataset))
                for _ in range(next_task_size):
                    pyg_chunks.append(dataset.pop())
                mpi_utils.node_comm.send((k, pyg_chunks), dest=worker)
                workers_free -= 1

        # Wait for all workers to finish
        logger.debug(f"Node root on {mpi_utils.hostname} signaling workers to quit.")
        while workers_free < mpi_utils.node_size - 1:
            data_t = mpi_utils.node_comm.recv(source=MPI.ANY_SOURCE, status=status)
            if data_t is not None:
                key, transformed_pyg_obj = data_t
                datasets_out[key].extend(transformed_pyg_obj)
            workers_free += 1

        # Tell all workers to terminate
        for worker in range(1, mpi_utils.node_size):
            mpi_utils.node_comm.send(None, dest=worker)

        t2 = time.time()
        logger.info(f"Graph transforms on {mpi_utils.hostname} done in {round(t2-t1)} seconds.")

        # write the transformed pyg objects
        t1 = time.time()
        write_adios_data(adios_out, datasets_out)
        t2 = time.time()
        if mpi_utils.node_roots_rank == 0:
            logger.info(f"ADIOS writing done in {round(t2-t1)} seconds.")

    except Exception as e:
        logger.error(f"Exception {e} at {traceback.format_exc()}")
        MPI.COMM_WORLD.Abort(1)


def _split_pyg_list(pyg_l, n):
    """
    Splits a list of PyG objects into n lists
    """
    sublists = list()
    listsize = len(pyg_l)
    k, m = divmod(listsize, n)

    for _ in range(n):
        l = []
        for i in range(k):
            obj = pyg_l.pop()
            l.append(obj)
        sublists.append(l)

    for i, _ in enumerate(range(m)):
        obj = pyg_l.pop()
        sublists[i].append(obj)

    return sublists
