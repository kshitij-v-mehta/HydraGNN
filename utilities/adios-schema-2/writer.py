import os, sys, pickle
from typing import List
from mpi4py import MPI
import numpy as np
from torch_geometric.data import Data
from hydragnn.utils.datasets import AdiosDataset
from hydragnn.utils.distributed import nsplit
from adios2 import Stream


def read_existing_dataset(filename, label, comm=MPI.COMM_WORLD):
    nproc = comm.Get_size()
    rank  = comm.Get_rank()

    dataset = AdiosDataset(filename, label, comm)
    rx = list(nsplit(range(len(dataset)), nproc))[rank]

    dataset.setsubset(rx[0], rx[-1]+1, preload=True)
    pyg_objects = [pyg for pyg in dataset]
    
    return pyg_objects


def _serialize_pyg_single(data: Data) -> np.ndarray:
    """Serialize a PyG Data object to a numpy uint8 array."""
    serialized_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(serialized_bytes, dtype=np.uint8).copy()


def _serialize_pyg_list(data_list: List[Data]):
    local_serialized = []
    for data in data_list:
        serialized = _serialize_pyg_single(data)
        local_serialized.append(serialized)

    return local_serialized


def _write_new_adios_schema(
    local_serialized: List[Data],
    label: str,
    stream: Stream,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> None:
    """
    Data layout in ADIOS2 file:
    - graph_data: Global 1D array containing all serialized graphs concatenated
    - graph_offsets: Global 1D array with starting offset of each graph in graph_data
    - graph_sizes: Global 1D array with byte size of each serialized graph
    
    Args:
        data_list: List of PyG Data objects local to this rank
        filename: Output ADIOS2 file path (e.g., "graphs.bp")
        comm: MPI communicator (defaults to MPI.COMM_WORLD)
    """
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_count = len(local_serialized)
    local_sizes = []
    for data in local_serialized:
        local_sizes.append(len(data))

    local_sizes = np.array(local_sizes, dtype=np.int64)
    local_total_bytes = int(np.sum(local_sizes))
    
    # Concatenate all local serialized data into one buffer
    if local_total_bytes > 0:
        local_data = np.concatenate(local_serialized)
    else:
        local_data = np.array([], dtype=np.uint8)
    
    # Gather counts and byte totals from all ranks
    all_counts = np.zeros(size, dtype=np.int64)
    all_bytes = np.zeros(size, dtype=np.int64)
    comm.Allgather(np.array([local_count], dtype=np.int64), all_counts)
    comm.Allgather(np.array([local_total_bytes], dtype=np.int64), all_bytes)
    
    total_graphs = int(np.sum(all_counts))
    total_bytes = int(np.sum(all_bytes))
    
    # Compute global offsets
    graph_offset = int(np.sum(all_counts[:rank]))  # Offset in graph index space
    byte_offset = int(np.sum(all_bytes[:rank]))    # Offset in byte space
    
    # Compute local graph offsets within the global data array
    local_offsets = np.zeros(local_count, dtype=np.int64)
    if local_count > 0:
        local_offsets[0] = byte_offset
        for i in range(1, local_count):
            local_offsets[i] = local_offsets[i-1] + local_sizes[i-1]
    
    # Write scalar metadata (only rank 0, but all ranks must participate)
    if rank == 0:
        stream.write(f"{label}/ndata", np.array([total_graphs], dtype=np.int64))
    
    # Write graph_data as a global distributed array
    # Each rank writes its portion: shape=[total], start=[byte_offset], count=[local_bytes]
    if total_bytes > 0:
        stream.write(
            "graph_data",
            local_data,
            shape=[total_bytes],
            start=[byte_offset],
            count=[local_total_bytes]
        )
    
    # Write graph_offsets as a global distributed array
    # Each rank writes its portion: shape=[total_graphs], start=[graph_offset], count=[local_count]
    if total_graphs > 0:
        stream.write(
            "graph_offsets",
            local_offsets,
            shape=[total_graphs],
            start=[graph_offset],
            count=[local_count]
        )
        
        # Write graph_sizes as a global distributed array
        stream.write(
            "graph_sizes",
            local_sizes,
            shape=[total_graphs],
            start=[graph_offset],
            count=[local_count]
        )
    
    if rank == 0:
        print(f"Successfully wrote {total_graphs} graphs ({total_bytes} bytes) from {size} ranks to {filename}")

    return total_graphs


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    assert len(sys.argv) == 3, f"Run as {sys.argv[0]} <input-adios-file> <output-adios_file>"
    filename = sys.argv[1]
    out_filename = sys.argv[2]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    total_graphs = 0
    
    with Stream(out_filename, "w", comm) as stream:
        for _ in stream.steps(1):
            for label in ("trainset", "testset", "valset"):
                pyg_objects = read_existing_dataset(filename, label)
                print(f"Found {len(pyg_objects)} in {label}")

                serialized_pyg = _serialize_pyg_list(pyg_objects)
                total_label_graphs = _write_new_adios_schema(serialized_pyg, label, stream)
                total_graphs += total_label_graphs

            if rank == 0:
                stream.write(f"ndata", np.array([total_graphs], dtype=np.int64))

