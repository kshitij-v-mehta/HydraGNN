import mpi_utils
import graphgps_transform


def node_worker(config):
    # Prepare the transform function
    ChemEncoder, lpe_transform = graphgps_transform.prepare_transform()

    # Keep accepting assignments from the node root till None is signalled
    while True:
        # Receive the next set of pyg objects from the node root
        pyg_object_list = mpi_utils.node_comm.recv(source=0)
        if pyg_object_list is None:
            break

        # Transform pyg objects one by one
        transformed_object_list = []
        while len(pyg_object_list) > 0:
            pyg_object = pyg_object_list.pop(0)  # pop to save memory
            transformed_pyg = graphgps_transform.graphgps_transform(ChemEncoder, lpe_transform, pyg_object, config)
            if transformed_pyg is not None:
                transformed_object_list.append(transformed_pyg)

        # Send the gps transformed objects to the node root
        mpi_utils.node_comm.send(transformed_object_list, dest=0)
