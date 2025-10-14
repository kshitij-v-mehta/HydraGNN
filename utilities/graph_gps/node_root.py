from adios_io import read_adios_data, write_adios_data
import mpi_utils
import traceback
from logger import logger
from mpi4py import MPI


def node_root(adios_in, adios_out):
    try:
        # Read adios data
        datasets_in = read_adios_data(adios_in)

        # Hold transformed pyg objects in a new dict
        datasets_out = dict()
        for k in datasets_in.keys():
            if k == 'extra_attrs':
                datasets_out[k] = datasets_in[k]
                continue
            datasets_out[k] = []

        # iterate over trainset, valset, and testset and send pyg objects to workers
        for k in datasets_in.keys():
            if k == 'extra_attrs': continue

            # Split list of PyG objects and send to workers
            dataset = datasets_in[k]
            pyg_chunks = _split_pyg_list(dataset, mpi_utils.node_size-1)
            for worker_rank in range(1, mpi_utils.node_size):
                pyg_chunk = pyg_chunks.pop()
                mpi_utils.node_comm.send(pyg_chunk, dest=worker_rank)

            # Receive list of transformed pyg objects from workers
            for worker_rank in range(1, mpi_utils.node_size):
                transformed_pyg_list = mpi_utils.node_comm.recv(source=worker_rank)
                datasets_out[k].extend(transformed_pyg_list)

        # signal workers to quit
        for worker_rank in range(1, mpi_utils.node_size):
            mpi_utils.node_comm.send(None, dest=worker_rank)

        # write the transformed pyg objects
        write_adios_data(adios_out, datasets_out)

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
