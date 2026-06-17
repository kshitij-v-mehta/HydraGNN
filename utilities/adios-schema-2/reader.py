import os, sys, pickle, time, traceback
import numpy as np
from torch_geometric.data import Data
from typing import List

from mpi4py import MPI
from adios2 import FileReader


def _deserialize_pyg_data(serialized: np.ndarray) -> Data:
    """Deserialize a numpy uint8 array back to a PyG Data object."""
    return pickle.loads(serialized.tobytes())


def _read_serialized_data(
    reader: FileReader,
    label: str,
    comm: MPI.Comm = None
) -> List[Data]:
    """
    Deserialize PyG Data objects in parallel from distributed global arrays.
    Each rank reads a subset of graphs.
    
    Optimized: reads all graph data for this rank in a single contiguous I/O operation.
    
    Args:
        filename: Input ADIOS2 file path
        comm: MPI communicator (defaults to MPI.COMM_WORLD)
    
    Returns:
        List of PyG Data objects local to this rank
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    data_list = []
    
    total_graphs = int(reader.read(f"{label}/ndata").item())
    
    if total_graphs == 0:
        return data_list
    
    # Distribute graphs evenly across ranks
    graphs_per_rank = total_graphs // size
    remainder = total_graphs % size
    
    if rank < remainder:
        local_count = graphs_per_rank + 1
        start_graph = rank * (graphs_per_rank + 1)
    else:
        local_count = graphs_per_rank
        start_graph = remainder * (graphs_per_rank + 1) + (rank - remainder) * graphs_per_rank
    
    if local_count == 0:
        return data_list
    
    # Read offsets and sizes for this rank's graphs (single I/O each)
    local_offsets = reader.read(
        f"{label}/graph_offsets",
        start=[start_graph],
        count=[local_count]
    )
    local_sizes = reader.read(
        f"{label}/graph_sizes",
        start=[start_graph],
        count=[local_count]
    )
    
    # Calculate contiguous byte range for all this rank's graphs
    byte_start = int(local_offsets[0])
    byte_end = int(local_offsets[-1]) + int(local_sizes[-1])
    total_local_bytes = byte_end - byte_start
    
    # Single I/O: read all graph data for this rank at once
    all_local_data = reader.read(
        f"{label}/graph_data",
        start=[byte_start],
        count=[total_local_bytes]
    )
    
    # Extract individual graphs from the contiguous buffer
    for i in range(local_count):
        # Offset relative to our local buffer
        rel_offset = int(local_offsets[i]) - byte_start
        size_bytes = int(local_sizes[i])
        
        serialized = all_local_data[rel_offset:rel_offset + size_bytes]
        data = _deserialize_pyg_data(serialized)
        data_list.append(data)
    
    if rank == 0:
        print(f"Successfully read {total_graphs} graphs across {size} ranks from {filename}")
    
    return data_list


#----------------------------------------------------------------------------#
if __name__ == '__main__':
    try:
        assert len(sys.argv) == 2, f"Run as {sys.argv[0]} <adios-file>"
        filename = sys.argv[1]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
 
        read_time = 0.0   
        with FileReader(filename, comm) as reader:
            for label in ("trainset", "valset", "testset"):
                t1 = time.time()
                pyg_obj_list = _read_serialized_data(reader, label)
                t2 = time.time()
                read_time += t2-t1
                print(f"Rank {rank} read {len(pyg_obj_list)} for {label}")

        if rank==0:
            print(f"read time: {read_time} seconds")

    except Exception as e:
        print(traceback.format_exc())

