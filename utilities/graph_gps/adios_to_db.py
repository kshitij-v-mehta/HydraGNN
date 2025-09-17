import json
import os.path
import pickle
import sys

from mpi4py import MPI

import hydragnn
from hydragnn.utils.datasets import AdiosDataset
from hydragnn.utils.distributed import nsplit
from utilities.graph_gps.db import DB


def parse_input_args():
    adios_in = sys.argv[1]
    db_dir_path = sys.argv[2]

    assert os.path.exists(adios_in), f"ADIOS input does not exist: {adios_in}"
    if not os.path.isdir(db_dir_path):
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(db_dir_path)

    return adios_in, db_dir_path


def open_db_connection(adios_in, db_dir_path, rank):
    dataset_name = os.path.basename(os.path.splitext(os.path.abspath(adios_in))[0])
    my_db_path = os.path.join(os.path.abspath(db_dir_path), f"{dataset_name}_{rank}.db")
    return DB(my_db_path)


def read_adios_data(adios_in, rank, nproc, comm=MPI.COMM_WORLD):
    common_variable_names = ["x", "edge_index", "edge_attr", "energy", "energy_per_atom", "forces", "pos", "y"]

    trainset = AdiosDataset(adios_in, "trainset", comm)
    valset = AdiosDataset(adios_in, "valset", comm)
    testset = AdiosDataset(adios_in, "testset", comm)

    for dataset in (trainset, valset, testset):
        rx = list(nsplit(range(len(dataset)), nproc))[rank]
        dataset.setkeys(common_variable_names)
        dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

    return trainset, valset, testset


def write_to_db(adios_datasets, db):
    for dataset in adios_datasets:
        for pyg in dataset:
            pickled_pyg = pickle.dumps(pyg)
            db.add(dataset.label, pickled_pyg)


def main():
    assert len(sys.argv) == 3, f"Run as {sys.argv[0]} <adios input> <path to db dir>"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    adios_in, db_dir_path = parse_input_args()
    db = open_db_connection(adios_in, db_dir_path, rank)
    trainset, valset, testset = read_adios_data(adios_in, rank, nproc, comm)
    write_to_db((trainset, valset, testset), db)


if __name__ == '__main__':
    main()
