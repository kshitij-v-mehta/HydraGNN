from mpi4py import MPI

from hydragnn.utils.datasets import AdiosWriter


def write_adios(filename, trainset, valset, testset):
    comm = MPI.COMM_WORLD
    # deg = gather_deg(trainset)

    adwriter = AdiosWriter(filename, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    adwriter.add_global("pna_deg", trainset.pna_deg)
    adwriter.add_global("dataset_name", trainset.dataset_name)
    adwriter.save()
