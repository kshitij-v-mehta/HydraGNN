"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
import copy
import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import cache, partial
from glob import glob
from pathlib import Path
from typing import Any, Callable

import ase
import numpy as np
from tqdm import tqdm

from fairchem.core.common.registry import registry
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.datasets.target_metadata_guesser import guess_property_metadata
from fairchem.core.modules.transforms import DataTransforms

################################################################

import sys

# Get the path relative to the script's location
this_dir = os.path.dirname(__file__)  # /project_root/scripts

sys.path.insert(0, this_dir)


######################################################################

from ase_db_backends.aselmdb import LMDBDatabase


def apply_one_tags(
    atoms: ase.Atoms, skip_if_nonzero: bool = True, skip_always: bool = False
):
    """
    This function will apply tags of 1 to an ASE atoms object.
    It is used as an atoms_transform in the datasets contained in this file.

    Certain models will treat atoms differently depending on their tags.
    For example, GemNet-OC by default will only compute triplet and quadruplet interactions
    for atoms with non-zero tags. This model throws an error if there are no tagged atoms.
    For this reason, the default behavior is to tag atoms in structures with no tags.

    args:
        skip_if_nonzero (bool): If at least one atom has a nonzero tag, do not tag any atoms

        skip_always (bool): Do not apply any tags. This arg exists so that this function can be disabled
                without needing to pass a callable (which is currently difficult to do with main.py)
    """
    if skip_always:
        return atoms

    if np.all(atoms.get_tags() == 0) or not skip_if_nonzero:
        atoms.set_tags(np.ones(len(atoms)))

    return atoms


class AseAtomsDataset(BaseDataset, ABC):
    """
    This is an abstract Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects. This should not be instantiated directly
    as get_atoms_object and load_dataset_get_ids are not implemented in this base class.

    Derived classes must add at least two things:
        self.get_atoms_object(id): a function that takes an identifier and returns a corresponding atoms object

        self.load_dataset_get_ids(config: dict): This function is responsible for any initialization/loads
            of the dataset and importantly must return a list of all possible identifiers that can be passed into
            self.get_atoms_object(id)

    Identifiers need not be any particular type.
    """

    def __init__(
        self,
        config: dict,
        atoms_transform: Callable[[ase.Atoms, Any, ...], ase.Atoms] = apply_one_tags,
    ) -> None:
        super().__init__(config)

        a2g_args = config.get("a2g_args", {}) or {}
        self.a2g = partial(AtomicData.from_ase, **a2g_args)

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

        self.atoms_transform = atoms_transform

        if self.config.get("keep_in_memory", False):
            self.__getitem__ = cache(self.__getitem__)

        self.ids = self._load_dataset_get_ids(config)
        self.num_samples = len(self.ids)

        if len(self.ids) == 0:
            raise ValueError(
                rf"No valid ase data found! \n"
                f"Double check that the src path and/or glob search pattern gives ASE compatible data: {config['src']}"
            )

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms(self.ids[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        sid = atoms.info.get("sid", self.ids[idx])

        fid = atoms.info.get("fid", None)
        if fid is not None:
            sid = f"{sid}-frame-{fid}"

        data_object = self.a2g(atoms, sid=sid)

        # Transform data object
        data_object = self.transforms(data_object)

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(data_object, self.key_mapping)

        if self.config.get("include_relaxed_energy", False):
            data_object.energy_relaxed = self.get_relaxed_energy(self.ids[idx])

        return data_object

    @abstractmethod
    def get_atoms(self, idx: str | int) -> ase.Atoms:
        # This function should return an ASE atoms object.
        raise NotImplementedError(
            "Returns an ASE atoms object. Derived classes should implement this function."
        )

    @abstractmethod
    def _load_dataset_get_ids(self, config):
        # This function should return a list of ids that can be used to index into the database
        raise NotImplementedError(
            "Every ASE dataset needs to declare a function to load the dataset and return a list of ids."
        )

    def get_relaxed_energy(self, identifier):
        raise NotImplementedError(
            "Reading relaxed energy from trajectory or file is not implemented with this dataset. "
            "If relaxed energies are saved with the atoms info dictionary, they can be used by passing the keys in "
            "the r_data_keys argument under a2g_args."
        )

    def sample_property_metadata(self, num_samples: int = 100) -> dict:
        metadata = {}

        if num_samples < len(self):
            metadata["targets"] = guess_property_metadata(
                [
                    self.get_atoms(self.ids[idx])
                    for idx in np.random.choice(
                        len(self), size=(num_samples,), replace=False
                    )
                ]
            )
        else:
            metadata["targets"] = guess_property_metadata(
                [self.get_atoms(self.ids[idx]) for idx in range(len(self))]
            )

        return metadata

    def get_metadata(self, attr, idx):
        # try the parent method
        metadata = super().get_metadata(attr, idx)
        if metadata is not None:
            return metadata
        # try to resolve it here
        if attr != "natoms":
            return None
        if isinstance(idx, (list, np.ndarray)):
            return np.array([self.get_metadata(attr, i) for i in idx])
        return len(self.get_atoms(idx))


@registry.register_dataset("ase_read")
class AseReadDataset(AseAtomsDataset):
    """
    This Dataset uses ase.io.read to load data from a directory on disk.
    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the efficiency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            pattern (str): Filepath matching each file you want to read
                    ex. "*/POSCAR", "*.cif", "*.xyz"
                    search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

            a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                    The relaxed structure is assumed to be the final structure in the file
                    (e.g. the last frame of a .traj).

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

            key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                in the model with the corresponding property as it was named in the dataset. Only need to use if
                the name is different.

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.
    """

    def _load_dataset_get_ids(self, config) -> list[Path]:
        self.ase_read_args = config.get("ase_read_args", {})

        if ":" in self.ase_read_args.get("index", ""):
            raise NotImplementedError(
                "To read multiple structures from a single file, please use AseReadMultiStructureDataset."
            )

        self.path = Path(config["src"])
        if self.path.is_file():
            raise ValueError(
                f"The specified src is not a directory: {self.config['src']}"
            )

        if self.config.get("include_relaxed_energy", False):
            self.relaxed_ase_read_args = copy.deepcopy(self.ase_read_args)
            self.relaxed_ase_read_args["index"] = "-1"

        return list(self.path.glob(f'{config.get("pattern", "*")}'))

    def get_atoms(self, idx: str | int) -> ase.Atoms:
        try:
            atoms = ase.io.read(idx, **self.ase_read_args)
        except Exception as err:
            warnings.warn(f"{err} occured for: {idx}", stacklevel=2)
            raise err

        return atoms

    def get_relaxed_energy(self, identifier) -> float:
        relaxed_atoms = ase.io.read(identifier, **self.relaxed_ase_read_args)
        return relaxed_atoms.get_potential_energy(apply_constraint=False)


@registry.register_dataset("ase_read_multi")
class AseReadMultiStructureDataset(AseAtomsDataset):
    """
    This Dataset can read multiple structures from each file using ase.io.read.
    The disadvantage is that all files must be read at startup.
    This is a significant cost for large datasets.

    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the efficiency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            pattern (str): Filepath matching each file you want to read
                    ex. "*.traj", "*.xyz"
                    search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

            index_file (str): Filepath to an indexing file, which contains each filename
                    and the number of structures contained in each file. For instance:

                    /path/to/relaxation1.traj 200
                    /path/to/relaxation2.traj 150

                    This will overrule the src and pattern that you specify!

            a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                    The relaxed structure is assumed to be the final structure in the file
                    (e.g. the last frame of a .traj).

            use_tqdm (bool): Use TQDM progress bar when initializing dataset

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

            key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                in the model with the corresponding property as it was named in the dataset. Only need to use if
                the name is different.

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
            object. Useful for applying tags, for example.

        transform (callable, optional): Additional preprocessing function for the Data object
    """

    def _load_dataset_get_ids(self, config) -> list[str]:
        self.ase_read_args = config.get("ase_read_args", {})
        if not hasattr(self.ase_read_args, "index"):
            self.ase_read_args["index"] = ":"

        if config.get("index_file", None) is not None:
            with open(config["index_file"]) as f:
                index = f.readlines()

            ids = []
            for line in index:
                filename = line.split(" ", maxsplit=1)[0]
                ids = [f"{filename} {i}" for i in range(int(line.split(" ")[1]))]

            return ids

        self.path = Path(config["src"])
        if self.path.is_file():
            raise ValueError(
                f"The specified src is not a directory: {self.config['src']}"
            )

        filenames = list(self.path.glob(f'{config.get("pattern", "*")}'))

        ids = []

        if config.get("use_tqdm", True):
            filenames = tqdm(filenames)
        for filename in filenames:
            try:
                structures = ase.io.read(filename, **self.ase_read_args)
            except Exception as err:
                warnings.warn(f"{err} occured for: {filename}", stacklevel=2)
            else:
                for i, _ in enumerate(structures):
                    ids.append(f"{filename} {i}")

        return ids

    def get_atoms(self, idx: str) -> ase.Atoms:
        try:
            identifiers = idx.split(" ")
            atoms = ase.io.read("".join(identifiers[:-1]), **self.ase_read_args)[
                int(identifiers[-1])
            ]
        except Exception as err:
            warnings.warn(f"{err} occured for: {idx}", stacklevel=2)
            raise err

        if "sid" not in atoms.info:
            atoms.info["sid"] = "".join(identifiers[:-1])
        if "fid" not in atoms.info:
            atoms.info["fid"] = int(identifiers[-1])

        return atoms

    def sample_property_metadata(self, num_samples: int = 100) -> dict:
        return {}

    def get_relaxed_energy(self, identifier) -> float:
        relaxed_atoms = ase.io.read(
            "".join(identifier.split(" ")[:-1]), **self.ase_read_args
        )[-1]
        return relaxed_atoms.get_potential_energy(apply_constraint=False)


@registry.register_dataset("ase_db")
class AseDBDataset(AseAtomsDataset):
    """
    This Dataset connects to an ASE Database, allowing the storage of atoms objects
    with a variety of backends including JSON, SQLite, and database server options.

    For more information, see:
    https://databases.fysik.dtu.dk/ase/ase/db/db.html

    args:
        config (dict):
            src (str): Either
                    - the path an ASE DB,
                    - the connection address of an ASE DB,
                    - a folder with multiple ASE DBs,
                    - a list of folders with ASE DBs
                    - a glob string to use to find ASE DBs, or
                    - a list of ASE db paths/addresses.
                    If a folder, every file will be attempted as an ASE DB, and warnings
                    are raised for any files that can't connect cleanly

                    Note that for large datasets, ID loading can be slow and there can be many
                    ids, so it's advised to make loading the id list as easy as possible. There is not
                    an obvious way to get a full list of ids from most ASE dbs besides simply looping
                    through the entire dataset. See the AseLMDBDataset which was written with this usecase
                    in mind.

            connect_args (dict): Keyword arguments for ase.db.connect()

            select_args (dict): Keyword arguments for ase.db.select()
                    You can use this to query/filter your database

            a2g_args (dict): Keyword arguments for fairchem.core.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True, "r_forces":True, and/or "r_stress":True as appropriate
                    In that case, energy/forces must be in the database

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transforms (dict[str, dict]): Dictionary specifying data transforms as {transform_function: config}
                    where config is a dictionary specifying arguments to the transform_function

            key_mapping (dict[str, str]): Dictionary specifying a mapping between the name of a property used
                in the model with the corresponding property as it was named in the dataset. Only need to use if
                the name is different.

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.

        transform (callable, optional): deprecated?
    """

    def _load_dataset_get_ids(self, config: dict) -> list[int]:
        if isinstance(config["src"], list):
            filepaths = []
            for path in sorted(config["src"]):
                if os.path.isdir(path):
                    filepaths.extend(sorted(glob(f"{path}/*")))
                elif os.path.isfile(path):
                    filepaths.append(path)
                else:
                    raise RuntimeError(f"Error reading dataset in {path}!")
        elif os.path.isfile(config["src"]):
            filepaths = [config["src"]]
        elif os.path.isdir(config["src"]):
            filepaths = sorted(glob(f'{config["src"]}/*'))
        else:
            filepaths = sorted(glob(config["src"]))

        self.dbs = []

        for path in filepaths:
            try:
                self.dbs.append(
                    self.connect_db(
                        path,
                        config.get("connect_args", {}),
                    )
                )
            except ValueError:
                logging.debug(
                    f"Tried to connect to {path} but it's not an ASE database!"
                )

        self.select_args = config.get("select_args", {})
        if self.select_args is None:
            self.select_args = {}

        # In order to get all of the unique IDs using the default ASE db interface
        # we have to load all the data and check ids using a select. This is extremely
        # inefficient for large dataset. If the db we're using already presents a list of
        # ids and there is no query, we can just use that list instead and save ourselves
        # a lot of time!
        self.db_ids = []
        for db in self.dbs:
            if hasattr(db, "ids") and self.select_args == {}:
                self.db_ids.append(db.ids)
            else:
                # this is the slow alternative
                self.db_ids.append([row.id for row in db.select(**self.select_args)])

        idlens = [len(ids) for ids in self.db_ids]
        self._idlen_cumulative = np.cumsum(idlens).tolist()

        return list(range(sum(idlens)))

    def get_atoms(self, idx: int) -> ase.Atoms:
        """Get atoms object corresponding to datapoint idx. Useful to read other properties not in data object.
        Args:
            idx (int): index in dataset

        Returns:
            atoms: ASE atoms corresponding to datapoint idx
        """
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._idlen_cumulative, idx)

        # Extract index of element within that db
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._idlen_cumulative[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()

        # put data back into atoms info
        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        return atoms

    @staticmethod
    def connect_db(
        address: str | Path, connect_args: dict | None = None
    ) -> ase.db.core.Database:
        if connect_args is None:
            connect_args = {}

        # If we're using an aselmdb, let's set readonly=True to be safe!
        if "aselmdb" in address:
            connect_args["readonly"] = True
            connect_args["use_lock_file"] = False

        return ase.db.connect(address, **connect_args)

    # def __del__(self):
    # for db in self.dbs:
    #    if hasattr(db, "close"):
    #        db.close()

    def sample_property_metadata(self, num_samples: int = 100) -> dict:
        logging.warning(
            "You specific a folder of ASE dbs, so it's impossible to know which metadata to use. Using the first!"
        )
        if self.dbs[0].metadata == {}:
            return super().sample_property_metadata(num_samples)

        return copy.deepcopy(self.dbs[0].metadata)
