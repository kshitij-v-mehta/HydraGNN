import mpi_utils
import graphgps_transform
import traceback
from mpi4py import MPI
from logger import logger


def node_worker(config):
    try:
        # Prepare the transform function
        ChemEncoder, lpe_transform = graphgps_transform.prepare_transform(config)

        # Send ping to indicate ready
        mpi_utils.node_comm.send(None, dest=0)

        # Keep accepting assignments from the node root till None is signalled
        while True:
            # Receive the next set of pyg objects from the node root
            data_t = mpi_utils.node_comm.recv(source=0)
            if data_t is None:
                break

            # Transform pyg objects one by one
            k, pyg_object_list = data_t
            logger.debug(f"Worker {mpi_utils.node_rank} on {mpi_utils.hostname} received {len(pyg_object_list)} "
                         f"objects from node root")

            transformed_object_list = []
            while len(pyg_object_list) > 0:
                pyg_object = pyg_object_list.pop()  # pop to save memory
                pyg_object = graphgps_transform.graphgps_transform(ChemEncoder, lpe_transform, pyg_object, config)
                if pyg_object is not None:
                    transformed_object_list.append(pyg_object)

            # Send the gps transformed objects to the node root
            mpi_utils.node_comm.send((k, transformed_object_list), dest=0)

        logger.info(f"Worker {mpi_utils.node_rank} on {mpi_utils.hostname} done. Goodbye.")

    except Exception as e:
        logger.error(f"Exception {e} at {traceback.format_exc()}")
        MPI.COMM_WORLD.Abort(1)
