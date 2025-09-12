import pdb
import sys, json
import torch
from mpi4py import MPI
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
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
    # Read data from ADIOS file
    trainset, valset, testset = read_adios(sys.argv[2])

    # Load config json
    jsonfile = sys.argv[1]
    with open(jsonfile, "r") as f:
        config = json.load(f)

    # Apply graph gps transform to every graph in the list
    for set_type in (trainset, valset, testset):
        for graph in set_type:
            graphgps_transform(graph, config)

    # Write data back to ADIOS file.
    write_adios(sys.argv[3], trainset, valset, testset)


if __name__ == "__main__":
    main()
