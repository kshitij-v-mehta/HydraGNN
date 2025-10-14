from adios_io import read_adios_data, write_adios_data
import mpi_utils
import numpy as np


def node_root(adios_in, adios_out):
    # Read adios data
    trainset, valset, testset, extra_attrs = read_adios_data(adios_in)

    # Hold transformed pyg objects
    datasets_out = {trainset.label: [], valset.label: [], testset.label: []}

    # Parse pyg objects
    for dataset in (trainset, valset, testset):

        # Split list and send tasks to workers
        pyg_chunks = np.array_split(dataset, mpi_utils.node_size-1)
        for worker_rank in range(1, mpi_utils.node_size):
            mpi_utils.node_comm.send(pyg_chunks, dest=worker_rank)

        # Receive list of transformed pyg objects from workers
        for worker_rank in range(1, mpi_utils.node_size):
            transformed_pyg_list = mpi_utils.node_comm.recv(source=worker_rank)
            datasets_out[dataset.label].append(transformed_pyg_list)

    write_adios_data(adios_out, datasets_out[trainset.label], datasets_out[valset.label], datasets_out[testset.label],
                     extra_attrs)
