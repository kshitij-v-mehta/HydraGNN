import glob
import json
import os
import pickle
import sys
import time

import torch
from fontTools.misc.psCharStrings import t2Operators
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


def write_adios_data(adios_out, trainset, valset, testset, comm):
    adwriter = AdiosWriter(adios_out, comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset", valset)
    adwriter.add("testset", testset)
    adwriter.add_global("pna_deg", trainset.pna_deg)
    adwriter.add_global("dataset_name", trainset.dataset_name)
    adwriter.save()


def prepare_transform():
    # Transformation to create positional and structural laplacian encoders
    # Chemical encoder
    ChemEncoder = ChemicalFeatureEncoder()

    # LPE
    lpe_transform = AddLaplacianEigenvectorPE(
        k=config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
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
                config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
            ],
            dtype=data.x.dtype,
            device=data.x.device,
        )

    data = ChemEncoder.compute_chem_features(data)
    data = compute_topo_features(data)
    return data


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert len(sys.argv) == 4, f"Run as {sys.argv[0]} <input adios file> <output adios file> <config file>"
    adios_in = sys.argv[1]
    adios_out = sys.argv[2]
    configfile = sys.argv[3]

    with open(configfile, "r") as f:
        config = json.load(f)

    ChemEncoder, lpe_transform = prepare_transform()

    if rank == 0:
        print("Reading adios data")
        t1 = time.time()

    trainset, valset, test = read_adios_data(adios_in, rank, size, comm)

    if rank == 0:
        t2 = time.time()
        print(f"Read adios data in {round(t2 - t1)} seconds")

    if rank == 0:
        print(f"Applying graph GPS transform")
        t1 = time.time()

    for dataset in (trainset, valset, test):
        for pyg in dataset:
            graphgps_transform(ChemEncoder, lpe_transform, pyg, config)

    if rank == 0:
        t2 = time.time()
        print(f"Done with graph GPS transform in {round(t2 - t1)} seconds")

    if rank == 0:
        print("Writing adios data")
        t1 = time.time()

    write_adios_data(adios_out, trainset, valset, test, comm)

    if rank == 0:
        t2 = time.time()
        print(f"Write adios data in {round(t2 - t1)} seconds")

    MPI.COMM_WORLD.Barrier()
    if rank == 0: print("All done. Goodbye.")
