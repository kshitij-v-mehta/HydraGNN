from mpi4py import MPI
import mpi_utils
from logger import logger
from hydragnn.utils.datasets import AdiosDataset, AdiosWriter
from adios2 import FileReader
from hydragnn.utils.distributed import nsplit


chunksize = int(1e5)


def read_adios_dataset(adios_in, label):
    """
    A generator that reads data from an adios file in chunks of size chunksize.
    label can be trainset, valset, or testset.
    """
    global chunksize
    comm = mpi_utils.node_roots_comm
    nproc = mpi_utils.node_roots_size
    rank = mpi_utils.node_roots_rank

    dataset = AdiosDataset(adios_in, label, comm)
    rx = list(nsplit(range(len(dataset)), nproc))[rank]
    start = rx[0]
    while start < rx[-1] + 1:
        end = min(start + chunksize, rx[-1] + 1)
        dataset.setsubset(start, end, preload=True)
        pyg_chunk = [pyg for pyg in dataset]
        yield pyg_chunk

        start += chunksize


def read_extra_attrs(adios_in):
    """
    Read extra attributes that may exist in the adios file
    """
    comm = mpi_utils.node_roots_comm
    rank = mpi_utils.node_roots_rank

    extra_attrs = dict()
    if rank == 0:
        with FileReader(adios_in) as f:
            attr = f.available_attributes()
            for a in attr.keys():
                if not any(
                        s in a for s in ["trainset/", "valset/", "testset/", "total_ndata"]
                ):
                    extra_attrs[a] = f.read_attribute(a)

    extra_attrs = comm.bcast(extra_attrs, root=0)
    return extra_attrs


def write_adios_data(adios_out, datasets):
    comm = mpi_utils.node_roots_comm

    adwriter = AdiosWriter(adios_out, comm)
    for k in datasets.keys():
        if k == 'extra_attrs': continue
        adwriter.add(k, datasets[k])

    attrs = datasets['extra_attrs']
    for a in attrs.keys():
        adwriter.add_global(a, attrs[a])
    adwriter.save()
