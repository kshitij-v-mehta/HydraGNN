from mpi4py import MPI
from adios_reader import read_adios_data
from adios_writer import write_to_adios
import mpi_utils
import numpy as np


def node_root():
    datasets_out = {'trainset': [], 'valset': [], 'testset': []}

    for dataset_type in ('trainset', 'valset', 'testset'):
        pyg_objects = read_adios_data(dataset_type, comm=mpi_utils.node_roots_comm)

        # Send tasks to workers
        pyg_chunks = np.array_split(pyg_objects, mpi_utils.node_size-1)
        for worker_rank in range(1, mpi_utils.node_size):
            mpi_utils.node_comm.send(pyg_chunks, dest=worker_rank)

        # Receive list of transformed pyg objects from workers
        for worker_rank in range(1, mpi_utils.node_size):
            transformed_pyg_list = mpi_utils.node_comm.recv(source=worker_rank)
            datasets_out[dataset_type].append(transformed_pyg_list)

    write_to_adios()
