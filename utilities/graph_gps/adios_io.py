import mpi_utils
from hydragnn.utils.datasets import AdiosDataset, AdiosWriter
from adios2 import FileReader
from hydragnn.utils.distributed import nsplit


def read_adios_data(adios_in):
    comm = mpi_utils.node_roots_comm
    nproc = mpi_utils.node_roots_size
    rank = mpi_utils.node_roots_rank

    trainset = AdiosDataset(adios_in, "trainset", comm)
    valset = AdiosDataset(adios_in, "valset", comm)
    testset = AdiosDataset(adios_in, "testset", comm)

    for dataset in (trainset, valset, testset):
        rx = list(nsplit(range(len(dataset)), nproc))[rank]
        print(f"Rank {rank} reading indices {rx[0]} to {rx[-1]} of {len(dataset)}")
        dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

    # Copy any additional attributes in the ADIOS file
    write_attrs = dict()
    if rank == 0:
        with FileReader(adios_in) as f:
            attr = f.available_attributes()
            for a in attr.keys():
                if not any(
                    s in a for s in ["trainset/", "valset/", "testset/", "total_ndata"]
                ):
                    write_attrs[a] = f.read_attribute(a)

    write_attrs = comm.bcast(write_attrs, root=0)
    return trainset, valset, testset, write_attrs


def write_adios_data(adios_out, trainset, valset, testset, attrs):
    comm = mpi_utils.node_roots_comm

    adwriter = AdiosWriter(adios_out, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    for a in attrs.keys():
        adwriter.add_global(a, attrs[a])
    adwriter.save()
