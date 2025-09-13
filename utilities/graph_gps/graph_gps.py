import sys, json
import time

import torch
from mpi4py import MPI
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import (
    ChemicalFeatureEncoder,
)
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import (
    compute_topo_features,
)

import hydragnn
from hydragnn.utils.datasets import AdiosDataset, AdiosWriter


def graphgps_transform(data, config):
    try:
        # Transformation to create positional and structural laplacian encoders
        # Chemical encoder
        ChemEncoder = ChemicalFeatureEncoder()

        # LPE
        lpe_transform = AddLaplacianEigenvectorPE(
            k=config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
            attr_name="lpe",
            is_undirected=True,
        )

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


def read_adios(filename):
    comm = MPI.COMM_WORLD

    trainset = AdiosDataset(filename, "trainset", comm)
    valset = AdiosDataset(filename, "valset", comm)
    testset = AdiosDataset(filename, "testset", comm)

    # (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    #     trainset, valset, testset, batch_size=64
    # )

    return trainset, valset, testset
    # return train_loader, val_loader, test_loader


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


def main():
    assert len(sys.argv) == 4, f"Run as {sys.argv[0]} <config file> <input adios file> <output adios file>"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read data from ADIOS file
    if rank == 0:
        print(f"Reading ADIOS file {sys.argv[2]}")
        t1 = time.time()
    trainset, valset, testset = read_adios(sys.argv[2])
    if rank == 0:
        t2 = time.time()
        print(f"Read ADIOS file successfully in {round(t2-t1,2)} seconds")

    # Load config json
    jsonfile = sys.argv[1]
    with open(jsonfile, "r") as f:
        config = json.load(f)

    # Apply graph gps transform to every graph in the list
    if rank == 0:
        print(f"Applying graph transforms")
        t1 = time.time()
    for set_type in (trainset, valset, testset):
        if rank == 0: print(f"Starting with {set_type.label}")
        for graph in set_type:
            graphgps_transform(graph, config)

    comm.Barrier()
    if rank == 0:
        t2 = time.time()
        print(f"Finished applying graph transforms in {round(t2-t1,2)} seconds. Now writing data to {sys.argv[3]}")
        t1 = time.time()

    # Write data back to ADIOS file.
    write_adios(sys.argv[3], trainset, valset, testset)
    if rank == 0:
        t2 = time.time()
        print(f"Wrote new ADIOS file successfully in {round(t2-t1,2)} seconds")

    if rank == 0: print('All done. Goodbye.')


if __name__ == "__main__":
    main()
