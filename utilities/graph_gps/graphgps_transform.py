import glob
import json
import os
import pickle
import sys
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
from utilities.graph_gps.db import DB


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


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert len(sys.argv) == 4, f"Run as {sys.argv[0]} <name of dataset> <path to config file> <path to db dir>"
    dataset_name = sys.argv[1]
    configfile = sys.argv[2]
    db_dir_path = sys.argv[3]

    # Read dataset config file
    with open(configfile, "r") as f:
        config = json.load(f)

    # Open connection to db
    if rank == 0:
        dbfiles = glob.glob(os.path.join(db_dir_path, "*.db"))
        assert len(dbfiles) == size, f"Run this with {len(dbfiles)} MPI ranks."

    mydbfile = os.path.join(db_dir_path, dataset_name + f"_{rank}.db")
    db = DB(mydbfile)

    # Process all pyg objects
    if rank == 0: t1 = time.time()
    while True:
        row_data = db.get_unprocessed()
        if row_data is None:
            break

        rowid, set_type, pyg_blob, _pyg_transformed = row_data
        pyg = pickle.loads(pyg_blob)
        print(f"Rank {rank} applying transform on row id {rowid}")
        pyg_transformed = graphgps_transform(pyg, config)
        pyg_transformed_blob = pickle.dumps(pyg_transformed)
        db.update_pyg_transformed(rowid, pyg_transformed_blob)
        print(f"Rank {rank} updated row id {rowid}")

    if rank == 0: print(f"Finished transforming {dataset_name} in {round(time.time()-t1,0)} seconds.")
    db.close()

    MPI.COMM_WORLD.Barrier()
    if rank == 0: print("All done. Goodbye.")
