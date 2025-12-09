"""
Parallel MPI program to serialize and write PyTorch Geometric Data objects to ADIOS2.
Uses distributed global arrays for efficient parallel I/O.

Each MPI rank writes its portion of a large global array containing all serialized graphs.

Uses ADIOS2 2.10+ high-level Python API with MPI support.

Requires: torch, torch_geometric, adios2 (version 2.10+), numpy, mpi4py

Install with:
    pip install torch torch_geometric adios2 mpi4py

Run with:
    mpirun -n 4 python pyg_to_adios2.py
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Optional

from mpi4py import MPI
from adios2 import Stream, FileReader


def _serialize_pyg_data(data: Data) -> np.ndarray:
    """Serialize a PyG Data object to a numpy uint8 array."""
    serialized_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(serialized_bytes, dtype=np.uint8).copy()


def _deserialize_pyg_data(serialized: np.ndarray) -> Data:
    """Deserialize a numpy uint8 array back to a PyG Data object."""
    return pickle.loads(serialized.tobytes())


def serialize_pyg_parallel(
    data_list: List[Data],
    filename: str,
    comm: MPI.Comm = None
) -> None:
    """
    Serialize PyG Data objects in parallel using distributed global arrays.
    
    Data layout in ADIOS2 file:
    - graph_data: Global 1D array containing all serialized graphs concatenated
    - graph_offsets: Global 1D array with starting offset of each graph in graph_data
    - graph_sizes: Global 1D array with byte size of each serialized graph
    
    Args:
        data_list: List of PyG Data objects local to this rank
        filename: Output ADIOS2 file path (e.g., "graphs.bp")
        comm: MPI communicator (defaults to MPI.COMM_WORLD)
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_count = len(data_list)
    
    # Serialize all local graphs and get their sizes
    local_serialized = []
    local_sizes = []
    for data in data_list:
        serialized = _serialize_pyg_data(data)
        local_serialized.append(serialized)
        local_sizes.append(len(serialized))
    
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
    
    with Stream(filename, "w", comm) as stream:
        for _ in stream.steps(1):
            # Write scalar metadata (only rank 0, but all ranks must participate)
            if rank == 0:
                stream.write("total_graphs", np.array([total_graphs], dtype=np.int64))
                stream.write("total_bytes", np.array([total_bytes], dtype=np.int64))
                stream.write("num_ranks", np.array([size], dtype=np.int64))
            
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


def deserialize_pyg_parallel(
    filename: str,
    comm: MPI.Comm = None
) -> List[Data]:
    """
    Deserialize PyG Data objects in parallel from distributed global arrays.
    Each rank reads a subset of graphs.
    
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
    
    with FileReader(filename, comm) as reader:
        total_graphs = int(reader.read("total_graphs").item())
        
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
        
        # Read the offsets and sizes for our graphs
        local_offsets = reader.read(
            "graph_offsets",
            start=[start_graph],
            count=[local_count]
        )
        local_sizes = reader.read(
            "graph_sizes",
            start=[start_graph],
            count=[local_count]
        )
        
        # Read each graph's data using its offset and size
        for i in range(local_count):
            offset = int(local_offsets[i])
            size_bytes = int(local_sizes[i])
            
            serialized = reader.read(
                "graph_data",
                start=[offset],
                count=[size_bytes]
            )
            
            data = _deserialize_pyg_data(serialized)
            data_list.append(data)
    
    if rank == 0:
        print(f"Successfully read {total_graphs} graphs across {size} ranks from {filename}")
    
    return data_list


def deserialize_pyg_serial(filename: str) -> List[Data]:
    """
    Deserialize all PyG Data objects from a file (serial read).
    Useful for post-processing or analysis on a single node.
    
    Args:
        filename: Input ADIOS2 file path
    
    Returns:
        List of all PyG Data objects
    """
    data_list = []
    
    with FileReader(filename) as reader:
        total_graphs = int(reader.read("total_graphs").item())
        
        if total_graphs == 0:
            return data_list
        
        # Read all offsets and sizes at once
        all_offsets = reader.read("graph_offsets")
        all_sizes = reader.read("graph_sizes")
        
        # Read each graph
        for i in range(total_graphs):
            offset = int(all_offsets[i])
            size_bytes = int(all_sizes[i])
            
            serialized = reader.read(
                "graph_data",
                start=[offset],
                count=[size_bytes]
            )
            
            data = _deserialize_pyg_data(serialized)
            data_list.append(data)
    
    print(f"Successfully read {total_graphs} graphs from {filename}")
    return data_list


def deserialize_pyg_serial_fast(filename: str) -> List[Data]:
    """
    Fast serial deserialization - reads entire data array at once.
    More memory intensive but faster for smaller datasets.
    
    Args:
        filename: Input ADIOS2 file path
    
    Returns:
        List of all PyG Data objects
    """
    data_list = []
    
    with FileReader(filename) as reader:
        total_graphs = int(reader.read("total_graphs").item())
        
        if total_graphs == 0:
            return data_list
        
        # Read everything at once
        all_data = reader.read("graph_data")
        all_offsets = reader.read("graph_offsets")
        all_sizes = reader.read("graph_sizes")
        
        # Extract each graph from the buffer
        for i in range(total_graphs):
            offset = int(all_offsets[i])
            size_bytes = int(all_sizes[i])
            
            serialized = all_data[offset:offset + size_bytes]
            data = _deserialize_pyg_data(serialized)
            data_list.append(data)
    
    print(f"Successfully read {total_graphs} graphs from {filename}")
    return data_list


# ============================================================================
# Low-level API version (adios2.bindings) for fine-grained control
# ============================================================================

def serialize_pyg_lowlevel_parallel(
    data_list: List[Data],
    filename: str,
    comm: MPI.Comm = None,
    engine_type: str = "BP5"
) -> None:
    """
    Serialize using low-level ADIOS2 bindings with distributed global arrays.
    """
    import adios2.bindings as adios2_bindings
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_count = len(data_list)
    
    # Serialize all local graphs
    local_serialized = []
    local_sizes = []
    for data in data_list:
        serialized = _serialize_pyg_data(data)
        local_serialized.append(serialized)
        local_sizes.append(len(serialized))
    
    local_sizes = np.array(local_sizes, dtype=np.int64)
    local_total_bytes = int(np.sum(local_sizes))
    
    if local_total_bytes > 0:
        local_data = np.concatenate(local_serialized)
    else:
        local_data = np.array([], dtype=np.uint8)
    
    # Gather counts and bytes
    all_counts = np.zeros(size, dtype=np.int64)
    all_bytes = np.zeros(size, dtype=np.int64)
    comm.Allgather(np.array([local_count], dtype=np.int64), all_counts)
    comm.Allgather(np.array([local_total_bytes], dtype=np.int64), all_bytes)
    
    total_graphs = int(np.sum(all_counts))
    total_bytes = int(np.sum(all_bytes))
    graph_offset = int(np.sum(all_counts[:rank]))
    byte_offset = int(np.sum(all_bytes[:rank]))
    
    # Compute local offsets
    local_offsets = np.zeros(local_count, dtype=np.int64)
    if local_count > 0:
        local_offsets[0] = byte_offset
        for i in range(1, local_count):
            local_offsets[i] = local_offsets[i-1] + local_sizes[i-1]
    
    # Initialize ADIOS2 with MPI
    adios = adios2_bindings.ADIOS(comm)
    io_obj = adios.DeclareIO("PyGWriter")
    io_obj.SetEngine(engine_type)
    
    writer = io_obj.Open(filename, adios2_bindings.Mode.Write)
    writer.BeginStep()
    
    # Define and write scalar metadata (rank 0 only)
    if rank == 0:
        total_var = io_obj.DefineVariable(
            "total_graphs",
            np.array([total_graphs], dtype=np.int64),
            [], [], [1]
        )
        writer.Put(total_var, np.array([total_graphs], dtype=np.int64))
        
        bytes_var = io_obj.DefineVariable(
            "total_bytes",
            np.array([total_bytes], dtype=np.int64),
            [], [], [1]
        )
        writer.Put(bytes_var, np.array([total_bytes], dtype=np.int64))
        
        ranks_var = io_obj.DefineVariable(
            "num_ranks",
            np.array([size], dtype=np.int64),
            [], [], [1]
        )
        writer.Put(ranks_var, np.array([size], dtype=np.int64))
    
    # Define global arrays with shape, start, count
    if total_bytes > 0 and local_total_bytes > 0:
        data_var = io_obj.DefineVariable(
            "graph_data",
            local_data,
            [total_bytes],      # Global shape
            [byte_offset],      # Start offset for this rank
            [local_total_bytes] # Count for this rank
        )
        writer.Put(data_var, local_data)
    
    if total_graphs > 0 and local_count > 0:
        offsets_var = io_obj.DefineVariable(
            "graph_offsets",
            local_offsets,
            [total_graphs],
            [graph_offset],
            [local_count]
        )
        writer.Put(offsets_var, local_offsets)
        
        sizes_var = io_obj.DefineVariable(
            "graph_sizes",
            local_sizes,
            [total_graphs],
            [graph_offset],
            [local_count]
        )
        writer.Put(sizes_var, local_sizes)
    
    writer.EndStep()
    writer.Close()
    
    if rank == 0:
        print(f"Successfully wrote {total_graphs} graphs ({total_bytes} bytes) using low-level API")


def deserialize_pyg_lowlevel_parallel(
    filename: str,
    comm: MPI.Comm = None,
    engine_type: str = "BP5"
) -> List[Data]:
    """
    Deserialize using low-level ADIOS2 bindings from distributed global arrays.
    """
    import adios2.bindings as adios2_bindings
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    data_list = []
    
    adios = adios2_bindings.ADIOS(comm)
    io_obj = adios.DeclareIO("PyGReader")
    io_obj.SetEngine(engine_type)
    
    reader = io_obj.Open(filename, adios2_bindings.Mode.Read)
    reader.BeginStep()
    
    # Read total graphs
    total_var = io_obj.InquireVariable("total_graphs")
    total_arr = np.zeros(1, dtype=np.int64)
    reader.Get(total_var, total_arr)
    reader.PerformGets()
    total_graphs = int(total_arr[0])
    
    if total_graphs == 0:
        reader.EndStep()
        reader.Close()
        return data_list
    
    # Distribute graphs
    graphs_per_rank = total_graphs // size
    remainder = total_graphs % size
    
    if rank < remainder:
        local_count = graphs_per_rank + 1
        start_graph = rank * (graphs_per_rank + 1)
    else:
        local_count = graphs_per_rank
        start_graph = remainder * (graphs_per_rank + 1) + (rank - remainder) * graphs_per_rank
    
    if local_count == 0:
        reader.EndStep()
        reader.Close()
        return data_list
    
    # Read offsets and sizes for our portion
    offsets_var = io_obj.InquireVariable("graph_offsets")
    offsets_var.SetSelection([[start_graph], [local_count]])
    local_offsets = np.zeros(local_count, dtype=np.int64)
    reader.Get(offsets_var, local_offsets)
    
    sizes_var = io_obj.InquireVariable("graph_sizes")
    sizes_var.SetSelection([[start_graph], [local_count]])
    local_sizes = np.zeros(local_count, dtype=np.int64)
    reader.Get(sizes_var, local_sizes)
    
    reader.PerformGets()
    
    # Read each graph's data
    data_var = io_obj.InquireVariable("graph_data")
    for i in range(local_count):
        offset = int(local_offsets[i])
        size_bytes = int(local_sizes[i])
        
        data_var.SetSelection([[offset], [size_bytes]])
        serialized = np.zeros(size_bytes, dtype=np.uint8)
        reader.Get(data_var, serialized)
        reader.PerformGets()
        
        data = _deserialize_pyg_data(serialized)
        data_list.append(data)
    
    reader.EndStep()
    reader.Close()
    
    if rank == 0:
        print(f"Successfully read {total_graphs} graphs using low-level API")
    
    return data_list


# ============================================================================
# Example usage - run with: mpirun -n 4 python pyg_to_adios2.py
# ============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each rank creates its own local graphs
    torch.manual_seed(42 + rank)
    
    local_graphs = []
    num_local_graphs = 3
    
    for i in range(num_local_graphs):
        num_nodes = 4 + rank + i
        num_edges = 2 * num_nodes
        
        data = Data(
            x=torch.randn(num_nodes, 16),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 8),
            y=torch.tensor([rank]),  # Label with rank for verification
            pos=torch.randn(num_nodes, 3)
        )
        local_graphs.append(data)
    
    if rank == 0:
        print(f"Running with {size} MPI ranks")
        print(f"Each rank has {num_local_graphs} graphs")
        print(f"Total graphs: {size * num_local_graphs}")
    
    comm.Barrier()
    print(f"Rank {rank}: Created {len(local_graphs)} graphs with sizes: "
          f"{[g.num_nodes for g in local_graphs]}")
    comm.Barrier()
    
    # -------------------------------------------------------------------------
    # Method 1: High-level parallel API with global arrays
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n--- Method 1: High-level API (Global Arrays) ---")
    comm.Barrier()
    
    output_file = "pyg_graphs_global.bp"
    serialize_pyg_parallel(local_graphs, output_file, comm)
    
    comm.Barrier()
    
    # Read back in parallel
    loaded_graphs = deserialize_pyg_parallel(output_file, comm)
    print(f"Rank {rank}: Loaded {len(loaded_graphs)} graphs")
    
    comm.Barrier()
    
    # -------------------------------------------------------------------------
    # Method 2: Serial read of parallel-written file (rank 0 only)
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n--- Method 2: Serial read (fast) ---")
        all_graphs = deserialize_pyg_serial_fast(output_file)
        print(f"Read all {len(all_graphs)} graphs on rank 0")
        
        # Verify by checking labels
        labels = [int(g.y.item()) for g in all_graphs]
        print(f"Graph labels (should show rank origin): {labels}")
    
    comm.Barrier()
    
    # -------------------------------------------------------------------------
    # Method 3: Low-level parallel API
    # -------------------------------------------------------------------------
    if rank == 0:
        print("\n--- Method 3: Low-level API (Global Arrays) ---")
    comm.Barrier()
    
    lowlevel_file = "pyg_graphs_global_lowlevel.bp"
    serialize_pyg_lowlevel_parallel(local_graphs, lowlevel_file, comm)
    
    loaded_lowlevel = deserialize_pyg_lowlevel_parallel(lowlevel_file, comm)
    print(f"Rank {rank}: Loaded {len(loaded_lowlevel)} graphs (low-level)")
    
    comm.Barrier()
    if rank == 0:
        print("\nDone!")

