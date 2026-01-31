import os, sys
from hydragnn.utils.datasets import AdiosDataset
from hydragnn.utils.distributed import nsplit
from adios2 import FileReader
from hydragnn.utils.datasets import AdiosDataset, AdiosWriter
from mpi4py import MPI
import torch

import logging

logger = logging.getLogger("remove_self_loops")
logging.basicConfig(
    format="%(levelname)s %(asctime)s %(message)s",
    level=os.environ.get("GPS_LOG_LEVEL", logging.DEBUG),
)


def should_skip_self_loops(data_object, context=""):
    """Return True if data_object contains self-loops in edge_index.

    Optionally prints a context string to help identify the source.
    """
    if not hasattr(data_object, "edge_index") or data_object.edge_index is None:
        return False
    loop_mask = data_object.edge_index[0] != data_object.edge_index[1]
    if torch.all(loop_mask):
        return False
    msg = "Skipping sample: self-loops detected in edge_index."
    if context:
        msg = f"{msg} Context: {context}"
    print(msg, flush=True)
    return True


def read_adios_dataset(adios_in, label, comm):
    nproc = comm.Get_size()
    rank = comm.Get_rank()

    logger.info(f"Reading {label}")
    dataset = AdiosDataset(adios_in, label, comm)
    rx = list(nsplit(range(len(dataset)), nproc))[rank]

    dataset.setsubset(rx[0], rx[-1] + 1, preload=True)
    pyg_objects = [pyg for pyg in dataset]

    logger.info(f"Done reading {label}")
    return pyg_objects


def read_extra_attrs(adios_in):
    """
    Read extra attributes that may exist in the adios file
    """
    extra_attrs = dict()

    with FileReader(adios_in) as f:
        attr = f.available_attributes()
        for a in attr.keys():
            if not any(s in a for s in ["trainset/", "valset/", "testset/", "total_ndata"]):
                extra_attrs[a] = f.read_attribute(a)

    return extra_attrs


def write_adios_data(adios_out, trainset, valset, testset, extra_attrs, comm):
    logger.info(f"Writing {adios_out}")
    adwriter = AdiosWriter(adios_out, comm)

    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)

    for k,v in extra_attrs.items():
        adwriter.add_global(k, v)
    adwriter.save()
    logger.info(f"Done writing {adios_out}")


def main():
    assert len(sys.argv) == 3, f"Run as {sys.argv[0]} <input_adios_file> <output_adios_file>"
    adios_in = sys.argv[1]
    adios_out = sys.argv[2]

    comm = MPI.COMM_WORLD

    trainset = read_adios_dataset(adios_in, "trainset", comm)
    logger.info(f"Filtering pyg objects with self loops")
    trainset_filtered = [pyg for pyg in trainset if not should_skip_self_loops(pyg)]

    valset = read_adios_dataset(adios_in, "valset", comm)
    logger.info(f"Filtering pyg objects with self loops")
    valset_filtered = [pyg for pyg in valset if not should_skip_self_loops(pyg)]

    testset = read_adios_dataset(adios_in, "testset", comm)
    logger.info(f"Filtering pyg objects with self loops")
    testset_filtered = [pyg for pyg in testset if not should_skip_self_loops(pyg)]

    logger.info("Done filtering all objects.")
    extra_attrs = read_extra_attrs(adios_in)

    write_adios_data(adios_out, trainset_filtered, valset_filtered, testset_filtered, extra_attrs, comm)
    logger.info("All done. Boodbye")


if __name__ == '__main__':
    main()
