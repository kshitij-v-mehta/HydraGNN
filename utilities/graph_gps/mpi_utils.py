"""
Create a set of MPI communicators:
    `node_comm` for all the processes on the same node
    `node_roots_comm` for all the root processes of nodes
    `node_workers_comm` for all the non-root worker processes on nodes

    Node roots ranks will participate in all adios reading and writing. That is, only one process from each node will
    perform I/O.

    After a node's root process reads PyG objects from adios, it will assign them to workers ranks on the node.
    Once all objects are processed, the root ranks will write data out to the adios file.

    This approach helps us adopt a hierarchical work distribution pattern, which helps manage large data efficiently.
    If a naive approach in which all ranks read their portion of data from adios, the metadata overhead is too high and
    the node runs out of memory.
"""

from mpi4py import MPI

global_comm = None
global_rank = None
global_size = None

node_comm = None
node_rank = None
node_size = None

node_roots_comm = None
node_roots_rank = None
node_roots_size = None

node_workers_comm = None


def _init():
    global global_comm, global_rank, global_size, node_comm, node_rank, node_size, node_roots_comm, node_roots_rank, \
        node_roots_size, node_workers_comm

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    node_comm = global_comm.Split_type(split_type=MPI.COMM_TYPE_SHARED, key=0)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()

    node_roots_comm = global_comm.Split(color=1 if node_rank == 0 else MPI.UNDEFINED, key=global_rank)
    node_roots_rank = node_roots_comm.Get_rank()
    node_roots_size = node_roots_comm.Get_size()

    node_workers_comm = global_comm.Split(color=1 if node_rank != 0 else MPI.UNDEFINED, key=global_rank)


# Initialize at import
_init()
