import os, json, getpass
import numpy as np
import torch
from mpi4py import MPI
import adios2.bindings as adios2
from torch_geometric.data import Data

N_STRUCTURES = 15000   # number of atomistic structures per rank
FLUSH_EVERY = 50   # flush to disk every N structures
dirpath = f"/mnt/bb/{getpass.getuser()}"


# --- Generate synthetic PyG structures ---

def make_structure(n_atoms=None):
    if n_atoms is None:
        n_atoms = np.random.randint(300, 500)

    pos    = torch.rand(n_atoms, 3, dtype=torch.float64) * 10.0
    z      = torch.randint(1, 119, (n_atoms,))
    forces = torch.randn(n_atoms, 3, dtype=torch.float64) * 0.1
    energy = torch.tensor(-3.5 * n_atoms + np.random.randn() * 0.5)

    return Data(pos=pos, z=z, forces=forces, energy=energy)


# --- ADIOS2 variable setup: define all variables once with dummy shape ---

def define_variables(io):
    """
    Define all variables once with a dummy shape of [1].
    SetShape + SetSelection will resize them per step.
    Energy is a fixed [1] scalar — no resizing needed.
    """
    dummy1d  = np.zeros(1, dtype=np.float64)
    dummy1di = np.zeros(1, dtype=np.int32)

    vars = {
        "atom_types":       io.DefineVariable("atom_types",       dummy1di, [1], [0], [1]),
        "coordinates_x":    io.DefineVariable("coordinates_x",    dummy1d,  [1], [0], [1]),
        "coordinates_y":    io.DefineVariable("coordinates_y",    dummy1d,  [1], [0], [1]),
        "coordinates_z":    io.DefineVariable("coordinates_z",    dummy1d,  [1], [0], [1]),
        "forces_x":         io.DefineVariable("forces_x",         dummy1d,  [1], [0], [1]),
        "forces_y":         io.DefineVariable("forces_y",         dummy1d,  [1], [0], [1]),
        "forces_z":         io.DefineVariable("forces_z",         dummy1d,  [1], [0], [1]),
        "formation_energy": io.DefineVariable("formation_energy", dummy1d,  [1], [0], [1]),
    }
    return vars


def write_adios_step(writer, vars, data):
    """Write one structure as one ADIOS2 step using SetShape + SetSelection."""
    atom_types       = data.z.numpy().astype(np.int32)
    coords_x         = data.pos[:, 0].numpy()
    coords_y         = data.pos[:, 1].numpy()
    coords_z         = data.pos[:, 2].numpy()
    forces_x         = data.forces[:, 0].numpy()
    forces_y         = data.forces[:, 1].numpy()
    forces_z         = data.forces[:, 2].numpy()
    formation_energy = np.array([data.energy.item()], dtype=np.float64)

    N = atom_types.shape[0]

    # Resize all per-atom variables to current N
    for name, arr in [
        ("atom_types",    atom_types),
        ("coordinates_x", coords_x),
        ("coordinates_y", coords_y),
        ("coordinates_z", coords_z),
        ("forces_x",      forces_x),
        ("forces_y",      forces_y),
        ("forces_z",      forces_z),
    ]:
        vars[name].SetShape([N])
        vars[name].SetSelection([[0], [N]])

    writer.BeginStep()
    writer.Put(vars["atom_types"],       atom_types)
    writer.Put(vars["coordinates_x"],    coords_x)
    writer.Put(vars["coordinates_y"],    coords_y)
    writer.Put(vars["coordinates_z"],    coords_z)
    writer.Put(vars["forces_x"],         forces_x)
    writer.Put(vars["forces_y"],         forces_y)
    writer.Put(vars["forces_z"],         forces_z)
    writer.Put(vars["formation_energy"], formation_energy)
    writer.EndStep()


# --- Periodic write to ADIOS2 ---

def write_adios(n_structures):
    filename = os.path.join(dirpath, f"structures-{MPI.COMM_WORLD.Get_rank()}.bp")
    a      = adios2.ADIOS()
    io     = a.DeclareIO("writer")
    io.SetEngine("BP5")
    writer = io.Open(filename, adios2.Mode.Write)
    vars   = define_variables(io)

    buffer = []
    for i in range(n_structures):
        buffer.append(make_structure())

        if len(buffer) >= FLUSH_EVERY:
            for data in buffer:
                write_adios_step(writer, vars, data)
            writer.PerformPuts()
            # print(f"  ADIOS flush: structures {i - len(buffer) + 1} to {i}")
            buffer.clear()

    if buffer:                                       # final partial flush
        for data in buffer:
            write_adios_step(writer, vars, data)
        writer.PerformPuts()
        # print(f"  ADIOS flush: final {len(buffer)} structures")
        buffer.clear()

    writer.Close()
    print(f"Done — wrote {n_structures} structures to {filename}\n")


# --- Periodic write to JSON ---

def write_json(n_structures):
    filename = os.path.join(dirpath, f"structures-{MPI.COMM_WORLD.Get_rank()}.json")
    buffer      = []
    first_flush = True

    with open(filename, "w") as f:
        f.write("[\n")

        for i in range(n_structures):
            buffer.append(make_structure())

            if len(buffer) >= FLUSH_EVERY:
                for j, data in enumerate(buffer):
                    record = {
                        "atom_types":       data.z.numpy().tolist(),
                        "coordinates_x":    data.pos[:, 0].numpy().tolist(),
                        "coordinates_y":    data.pos[:, 1].numpy().tolist(),
                        "coordinates_z":    data.pos[:, 2].numpy().tolist(),
                        "forces_x":         data.forces[:, 0].numpy().tolist(),
                        "forces_y":         data.forces[:, 1].numpy().tolist(),
                        "forces_z":         data.forces[:, 2].numpy().tolist(),
                        "formation_energy": data.energy.item(),
                    }
                    prefix = "" if (first_flush and j == 0) else ",\n"
                    f.write(prefix + json.dumps(record))
                f.flush()
                print(f"  JSON  flush: structures {i - len(buffer) + 1} to {i}")
                first_flush = False
                buffer.clear()

        if buffer:                                   # final partial flush
            for j, data in enumerate(buffer):
                record = {
                    "atom_types":       data.z.numpy().tolist(),
                    "coordinates_x":    data.pos[:, 0].numpy().tolist(),
                    "coordinates_y":    data.pos[:, 1].numpy().tolist(),
                    "coordinates_z":    data.pos[:, 2].numpy().tolist(),
                    "forces_x":         data.forces[:, 0].numpy().tolist(),
                    "forces_y":         data.forces[:, 1].numpy().tolist(),
                    "forces_z":         data.forces[:, 2].numpy().tolist(),
                    "formation_energy": data.energy.item(),
                }
                prefix = "" if (first_flush and j == 0) else ",\n"
                f.write(prefix + json.dumps(record))
            f.flush()
            print(f"  JSON  flush: final {len(buffer)} structures")
            buffer.clear()

        f.write("\n]")

    print(f"Done — wrote {n_structures} structures to {filename}\n")


# --- Read back and verify ---

def read_adios():
    filename = os.path.join(dirpath, f"structures-{MPI.COMM_WORLD.Get_rank()}.bp")
    a      = adios2.ADIOS()
    io     = a.DeclareIO("reader")
    reader = io.Open(filename, adios2.Mode.Read)

    print(f"Reading {filename}:")
    step = 0
    while reader.BeginStep() == adios2.StepStatus.OK:
        v_at = io.InquireVariable("atom_types")
        v_en = io.InquireVariable("formation_energy")

        N      = v_at.Shape()[0]
        energy = np.zeros(1, dtype=np.float64)

        reader.Get(v_en, energy)
        reader.EndStep()
        print(f"  [{step:3d}] natoms={N}  formation_energy={energy[0]:.3f} eV")
        step += 1

    reader.Close()


def read_json():
    filename = os.path.join(dirpath, f"structures-{MPI.COMM_WORLD.Get_rank()}.json")
    with open(filename) as f:
        records = json.load(f)

    print(f"\nReading {filename}:")
    for i, r in enumerate(records):
        print(f"  [{i:3d}] natoms={len(r['atom_types'])}  formation_energy={r['formation_energy']:.3f} eV")


# --- Main ---

if __name__ == "__main__":
    np.random.seed(42)

    print(f"=== Writing {N_STRUCTURES} structures, flushing every {FLUSH_EVERY} ===\n")
    write_adios(N_STRUCTURES)
    # write_json(N_STRUCTURES)

    # read_adios()
    # read_json()

