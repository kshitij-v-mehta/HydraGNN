from mpi4py import MPI
import mpi_utils
import sys, json
import hydragnn
from node_root import node_root
from node_worker import node_worker


def main():
    hydragnn.utils.print.setup_log("graphgps_transform")

    # Read input arguments
    assert (len(sys.argv) == 4), \
        f"Run as {sys.argv[0]} <input adios file> <output adios file> <config file>"
    adios_in = sys.argv[1]
    adios_out = sys.argv[2]
    configfile = sys.argv[3]

    with open(configfile, "r") as f:
        config = json.load(f)

    # Start node-local root and worker processes
    if mpi_utils.node_rank == 0:
        node_root(adios_in, adios_out)
    else:
        node_worker(config)


if __name__ == '__main__':
    main()
