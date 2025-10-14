import json
import os
import sys
import time
import traceback
from functools import partial

import torch
from mpi4py import MPI
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from hydragnn.utils.datasets import AdiosDataset, AdiosWriter
from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import (
    ChemicalFeatureEncoder,
)
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import (
    compute_topo_features,
)
from hydragnn.utils.distributed import nsplit

from adios2 import FileReader

import hydragnn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_adios_data(adios_in, rank, nproc, comm=MPI.COMM_WORLD):
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


def write_adios_data(adios_out, trainset, valset, testset, attrs, comm):
    adwriter = AdiosWriter(adios_out, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    for a in attrs.keys():
        adwriter.add_global(a, attrs[a])
    adwriter.save()


def prepare_transform():
    # Transformation to create positional and structural laplacian encoders
    # Chemical encoder
    ChemEncoder = ChemicalFeatureEncoder()

    # LPE
    lpe_transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"].get("num_laplacian_eigs", 5),  # return a default value of 5 if key not found
        attr_name="lpe",
        is_undirected=True,
    )

    return ChemEncoder, lpe_transform


def graphgps_transform(ChemEncoder, lpe_transform, data, config):
    try:
        data = lpe_transform(data)  # lapPE

    except:
        data.lpe = torch.zeros(
            [
                data.num_nodes,
                config["NeuralNetwork"]["Architecture"].get("num_laplacian_eigs", 5),
            ],
            dtype=data.x.dtype,
            device=data.x.device,
        )

    data = ChemEncoder.compute_chem_features(data)
    data = compute_topo_features(data)
    return data


def transform_one(i, dataset, ChemEncoder, lpe_transform, config):
    """
    For use with ThreadPoolExecutor
    """
    try:
        data = graphgps_transform(ChemEncoder=ChemEncoder, lpe_transform=lpe_transform, data=dataset[i], config=config)
        return data
    except Exception as e:
        print(f"Encountered non-fatal exception {e} at {traceback.format_exc()}. Ignoring data sample")
        # RuntimeError may occur if number of atoms is less than the number of eigenvectors. In that case,
        # ignore this dataset sample
        pass
        sys.exit(1)


if __name__ == "__main__":

    ## Set up logging
    hydragnn.utils.print.setup_log("graphgps_transform")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert (
        len(sys.argv) == 4
    ), f"Run as {sys.argv[0]} <input adios file> <output adios file> <config file>"
    adios_in = sys.argv[1]
    adios_out = sys.argv[2]
    configfile = sys.argv[3]

    with open(configfile, "r") as f:
        config = json.load(f)

    ChemEncoder, lpe_transform = prepare_transform()

    if rank == 0:
        print("Reading adios data")
        t1 = time.time()

    trainset, valset, testset, attrs = read_adios_data(adios_in, rank, size, comm)

    if rank == 0:
        t2 = time.time()
        print(f"Read adios data in {round(t2 - t1)} seconds")

    if rank == 0:
        print(f"Applying graph GPS transform")
        t1 = time.time()

    datasets_out = {trainset.label: [], valset.label: [], testset.label: []}
    for dataset in (trainset, valset, testset):
        with ProcessPoolExecutor() as executor:
            transform_func = partial(transform_one, dataset=dataset, ChemEncoder=ChemEncoder,
                                    lpe_transform=lpe_transform, config=config)
            futures = [executor.submit(transform_one, i, dataset=dataset, ChemEncoder=ChemEncoder,
                                       lpe_transform=lpe_transform, config=config) for i in range(len(dataset))]

            for fut in as_completed(futures):
                try:
                    data = fut.result()
                    if data is not None:
                        datasets_out[dataset.label].append(data)
                except:
                    pass

    if rank == 0:
        t2 = time.time()
        print(f"Done with graph GPS transform in {round(t2 - t1)} seconds")

    if rank == 0:
        print("Writing adios data")
        t1 = time.time()

    write_adios_data(adios_out, datasets_out['trainset'], datasets_out['valset'], datasets_out['testset'], attrs, comm)

    if rank == 0:
        t2 = time.time()
        print(f"Write adios data in {round(t2 - t1)} seconds")

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print("All done. Goodbye.")
