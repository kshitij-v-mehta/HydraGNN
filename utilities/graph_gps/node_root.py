from adios_io import read_adios_dataset, write_adios_data
import mpi_utils
import traceback
from logger import logger
from mpi4py import MPI
import time

from utilities.graph_gps.adios_io import read_extra_attrs


def node_root(adios_in, adios_out):
    try:
        # Hold transformed pyg objects in a new dict
        datasets_out = {'trainset': [], 'valset': [], 'testset': []}

        t1 = time.time()
        for label in ('trainset', 'valset', 'testset'):
            workers_free = 0
            status = MPI.Status()

            # iterate over trainset, valset, and testset and send pyg objects to workers
            logger.debug(f"Node root on {mpi_utils.hostname} sending tasks in {label} to workers")
            for pyg_chunk in read_adios_dataset(adios_in, label):
                logger.debug(f"Node root on {mpi_utils.hostname} read next chunk")
                task_size = len(pyg_chunk) // (mpi_utils.node_size * 2)

                # chop into tasks and dynamically assign them to workers
                while len(pyg_chunk) > 0:
                    # get ping/data from worker
                    data_t = mpi_utils.node_comm.recv(source=MPI.ANY_SOURCE, status=status)
                    if data_t is not None:
                        key, transformed_pyg_obj = data_t
                        datasets_out[key].extend(transformed_pyg_obj)
                    else:
                        pass  # this was a ready ping from a worker
                    worker = status.Get_source()
                    workers_free += 1

                    # send a chunk of tasks to worker
                    task_chunk = []
                    for _ in range(task_size):
                        if len(pyg_chunk) == 0:
                            break
                        task_chunk.append(pyg_chunk.pop())

                    if len(task_chunk) > 0:
                        mpi_utils.node_comm.send((label, task_chunk), dest=worker)
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

        datasets_out['extra_attrs'] = read_extra_attrs(adios_in)
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
