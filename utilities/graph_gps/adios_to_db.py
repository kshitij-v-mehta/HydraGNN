import os.path
import sys

from mpi4py import MPI

from hydragnn.utils.datasets import AdiosDataset
from utilities.graph_gps.db import DB


def main():
    assert len(sys.argv) == 3, f"Run as {sys.argv[0]} <adios input> <path to db dir>"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    adios_in = sys.argv[1]
    db_dir_path = sys.argv[2]

    assert os.path.exists(adios_in), f"ADIOS input does not exist: {adios_in}"
    if not os.path.isdir(db_dir_path):
        if rank == 0:
            os.makedirs(db_dir_path)

    dataset_name = os.path.basename(os.path.splitext(adios_in))
    my_db_path = os.path.join(db_dir_path, f"{dataset_name}_{rank}.db")
    db = DB(my_db_path)

    trainset = AdiosDataset(adios_in, "trainset", comm)
    valset = AdiosDataset(adios_in, "valset", comm)
    testset = AdiosDataset(adios_in, "testset", comm)

    for set_type in (trainset, testset, valset):
        for graphobj in set_type:
            db.add(set_type, graphobj)


if __name__ == '__main__':
    main()
