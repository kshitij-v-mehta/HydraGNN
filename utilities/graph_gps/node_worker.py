import mpi_utils
import graphgps_transform
import traceback
from mpi4py import MPI
from logger import logger


def node_worker(config):
    try:
        # Prepare the transform function
        ChemEncoder, lpe_transform = graphgps_transform.prepare_transform(config)

        # Keep accepting assignments from the node root till None is signalled
        while True:
            # Receive the next set of pyg objects from the node root
            pyg_object_list = mpi_utils.node_comm.recv(source=0)
            if pyg_object_list is None:
                break

            # Transform pyg objects one by one
            transformed_object_list = []
            while len(pyg_object_list) > 0:
                pyg_object = pyg_object_list.pop()  # pop to save memory
                transformed_pyg = graphgps_transform.graphgps_transform(ChemEncoder, lpe_transform, pyg_object, config)
                if transformed_pyg is not None:
                    transformed_object_list.append(transformed_pyg)

            # Send the gps transformed objects to the node root
            mpi_utils.node_comm.send(transformed_object_list, dest=0)

    except Exception as e:
        logger.error(f"Exception {e} at {traceback.format_exc()}")
        MPI.COMM_WORLD.Abort(1)
