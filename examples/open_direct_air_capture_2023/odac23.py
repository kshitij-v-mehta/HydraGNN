import os, random, torch, glob, sys, pickle, shutil, traceback, pdb, tqdm
from pickle import HIGHEST_PROTOCOL

import numpy as np
from joblib.externals.loky import ProcessPoolExecutor
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from networkx.readwrite.json_graph.node_link import node_link_data
from yaml import full_load
import time
import threading, queue
from sqlite_helper import DB

from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from torch_geometric.transforms import Distance, Spherical, LocalCartesian
from torch_geometric.data import Data
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    PBCLocalCartesian,
    pbc_as_tensor,
    gather_deg,
)
from hydragnn.utils.distributed import nsplit
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager

# transform_coordinates = Spherical(norm=False, cat=False)
transform_coordinates = LocalCartesian(norm=False, cat=False)
# transform_coordinates = Distance(norm=False, cat=False)

transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
# transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


import torch
from torch_geometric.data import Data, Dataset
from ase.io import read
from ase import Atoms
import os
from typing import List

from hydragnn.utils.print.print_utils import iterate_tqdm, log


class ExtendedXYZDataset(Dataset):
    def __init__(self, extxyz_filename: str, transform=None, pre_transform=None):
        """
        Args:
            file_list (List[str]): List of paths to `.extxyz` files.
        """
        super().__init__(None, transform, pre_transform)
        self.extxyz_filename = extxyz_filename
        self.structures = []
        self._load_structures()

    def _load_structures(self):
        """Reads all structures from all .extxyz files and stores them in self.structures"""
        if not os.path.isfile(self.extxyz_filename):
            raise FileNotFoundError(f"File not found: {self.extxyz_filename}")
        frames = read(self.extxyz_filename, index=':', parallel=False)  # Read all structures in file
        self.structures.extend(frames)

    def len(self):
        return len(self.structures)

    def get(self, idx):
        atoms: Atoms = self.structures[idx]

        return atoms


class ODAC2023(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        data_type,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
        stage_db=False,
        comm=MPI.COMM_WORLD,
    ):
        try:
            super().__init__()

            assert (data_type == "train") or (
                data_type == "val"
            ), "data_type must be a string either equal to 'train' or to 'val'"

            self.config = config
            self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
            self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

            self.data_path = os.path.join(dirpath, data_type)
            self.energy_per_atom = energy_per_atom

            self.radius_graph = RadiusGraph(
                self.radius, loop=False, max_num_neighbors=self.max_neighbours
            )
            self.radius_graph_pbc = RadiusGraphPBC(
                self.radius, loop=False, max_num_neighbors=self.max_neighbours
            )

            self.graphgps_transform = graphgps_transform

            # Threshold for atomic forces in eV/angstrom
            self.forces_norm_threshold = 1000.0

            self.dist = dist
            if self.dist:
                assert torch.distributed.is_initialized()
                self.world_size = torch.distributed.get_world_size()
                self.rank = torch.distributed.get_rank()
            self.comm = comm

            if stage_db:
                # Write pyg objects to db and return

                # First level MPI parallelization
                # Distribute unprocesses ext files amongst MPI ranks
                # Batch files into sets of w files so that they can be spread amongst processes on a node
                extfiles = self.get_unprocessed_extfiles(dirpath, data_type)

                if self.rank == 0:
                    status = MPI.Status()

                    # dynamically assign a file to a worker process
                    while len(extfiles) > 0:
                        self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                        worker = status.Get_source()
                        self.comm.send(extfiles.pop(), dest=worker, tag=worker)

                    # send terminate to workers as all files have been processed
                    for _ in range(1, self.world_size):
                        self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                        worker = status.Get_source()
                        self.comm.send("DONE", dest=worker, tag=worker)

                else:
                    # worker process
                    while True:
                        self.comm.send(self.rank, dest=0, tag=0)
                        extfile = self.comm.recv(source=0, tag=self.rank)
                        if extfile == "DONE":
                            break
                        self.process_file(extfile, dirpath, data_type)

            else:
                # Read pyg objects from db
                pass

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            MPI.COMM_WORLD.Abort(1)

    def process_file(self, extfile, dirpath, data_type):
        """
        Second-level MPI parallelization.
        MPI rank on each node runs this function.
        It takes in an ext file and spawns processes locally to process samples in the file in parallel.
        """
        path_id = extfile.removeprefix(dirpath).lstrip("/").replace("/","-")
        tmp_pyg_fname = f"/tmp/{path_id}.pt"

        t1 = time.time()
        try:
            dataset = ExtendedXYZDataset(extxyz_filename=extfile)
        except ValueError as e:
            print(f"{extfile} not a valid ase lmdb dataset. Ignoring ...")
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        if data_type == "train":
            rx = list(range(len(dataset)))
        else:
            rx = list(nsplit(list(range(len(dataset))), self.world_size))[self.rank]

        print(f"Rank: {self.rank}, dataname: {extfile}, data_type: {data_type}, "
              f"num_samples: {len(dataset)}, len(rx): {len(rx)}", flush=True)

        # for index in iterate_tqdm(
        #     rx,
        #     verbosity_level=2,
        #     desc=f"Rank{self.rank} Dataset {index}/{len(rx)}",
        # ):
        for index in rx:
            self._create_pytorch_data_object(dataset, index)
        
        torch.save(self.dataset, tmp_pyg_fname)
        os.rename(tmp_pyg_fname, f"{tmp_pyg_fname}-DONE")
        t2 = time.time()
        self.dataset = []
        print(f"Rank {self.rank} done processing {extfile} in {round(t2-t1,2)} seconds", flush=True)

    def get_unprocessed_extfiles(self, dirpath, data_type):
        """
        Get a list of ext files that have not been processed.
        """
        unprocessed_files = []
        if self.rank == 0:
            allextfiles = glob.glob(os.path.join(dirpath, data_type, "**/*.extxyz"))
            print(f"{len(allextfiles)} ext files found at {os.path.join(dirpath, data_type)}")
            db_files = glob.glob("odac23_*.db")
            print(f"Found {len(db_files)} db files")

            extfiles_done = list()
            for dbfile in db_files:
                dbm = DB(dbfile)
                extfiles_done.extend(dbm.get_all_filenames())

            extfiles_done = [os.path.join(dirpath, fname) for fname in extfiles_done]
            unprocessed_files = list(set(allextfiles) - set(extfiles_done))
            print(f"{len(unprocessed_files)} ext files to be processed", flush=True)

        return unprocessed_files

    def _create_pytorch_data_object(self, dataset, index):
        try:
            pos = torch.tensor(
                dataset.get(index).get_positions(), dtype=torch.float32
            )
            natoms = torch.IntTensor([pos.shape[0]])
            atomic_numbers = torch.tensor(
                dataset.get(index).get_atomic_numbers(),
                dtype=torch.float32,
            ).unsqueeze(1)

            energy = torch.tensor(
                dataset.get(index).get_total_energy(), dtype=torch.float32
            ).unsqueeze(0)

            energy_per_atom = energy.detach().clone() / natoms
            forces = torch.tensor(
                dataset.get(index).get_forces(), dtype=torch.float32
            )

            chemical_formula = dataset.get(index).get_chemical_formula()

            cell = None
            try:
                # cell = torch.tensor(
                #     dataset.get(index).get_cell(), dtype=torch.float32
                # ).view(3, 3)
                cell = torch.from_numpy(
                    np.asarray(dataset.get(index).get_cell())
                ).to(
                    torch.float32
                )  # dtype conversion in-place
                # shape is already (3, 3) so no .view needed
            except:
                print(
                    f"Atomic structure {chemical_formula} does not have cell",
                    flush=True,
                )

            pbc = None
            try:
                pbc = pbc_as_tensor(dataset.get(index).get_pbc())
            except:
                print(
                    f"Atomic structure {chemical_formula} does not have pbc",
                    flush=True,
                )

            # If either cell or pbc were not read, we set to defaults which are not none.
            if cell is None or pbc is None:
                cell = torch.eye(3, dtype=torch.float32)
                pbc = torch.tensor([False, False, False], dtype=torch.bool)

            x = torch.cat([atomic_numbers, pos, forces], dim=1)

            # Calculate chemical composition
            atomic_number_list = atomic_numbers.tolist()
            assert len(atomic_number_list) == natoms
            ## 118: number of atoms in the periodic table
            hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
            chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

            data_object = Data(
                dataset_name="odac23",
                natoms=natoms,
                pos=pos,
                cell=cell,
                pbc=pbc,
                edge_index=None,
                edge_attr=None,
                atomic_numbers=atomic_numbers,
                chemical_composition=chemical_composition,
                smiles_string=None,
                x=x,
                energy=energy,
                energy_per_atom=energy_per_atom,
                forces=forces,
            )

            if self.energy_per_atom:
                data_object.y = data_object.energy_per_atom
            else:
                data_object.y = data_object.energy

            if data_object.pbc.any():
                try:
                    data_object = self.radius_graph_pbc(data_object)
                    data_object = transform_coordinates_pbc(data_object)
                except:
                    print(
                        f"Structure could not successfully apply one or both of the pbc radius graph and positional transform",
                        flush=True,
                    )
                    data_object = self.radius_graph(data_object)
                    data_object = transform_coordinates(data_object)
            else:
                data_object = self.radius_graph(data_object)
                data_object = transform_coordinates(data_object)

            # Default edge_shifts for when radius_graph_pbc is not activated
            if not hasattr(data_object, "edge_shifts"):
                data_object.edge_shifts = torch.zeros(
                    (data_object.edge_index.size(1), 3), dtype=torch.float32
                )

            # FIXME: PBC from bool --> int32 to be accepted by ADIOS
            data_object.pbc = data_object.pbc.int()

            # LPE
            if self.graphgps_transform is not None:
                data_object = self.graphgps_transform(data_object)

            if self.check_forces_values(data_object.forces):
                self.dataset.append(data_object)
            else:
                print(
                    f"L2-norm of force tensor is {data_object.forces.norm()} and exceeds threshold {self.forces_norm_threshold} - atomistic structure: {chemical_formula}",
                    flush=True,
                )

        except Exception as e:
            print(f"Rank {self.rank} reading - exception: ", e)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]
