import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE


from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import (
    ChemicalFeatureEncoder,
)
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import (
    compute_topo_features,
)


def prepare_transform(config):
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
