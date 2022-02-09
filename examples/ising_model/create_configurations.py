#!/usr/bin/env python3
import os
import shutil
import numpy as np
from tqdm import tqdm
from sympy.utilities.iterables import multiset_permutations
import scipy.special
import math
from mpi4py import MPI
import adios2


class AdiosWriter_1:
    """
    This implementation creates a BP file in which total_energy, nodal_input, and nodal_output
    are variables. We write an adios step for each graph.
    Pros: A compact structure that may have low metadata overhead.
    Cons: Cannot do data parallelism as it cannot write graphs (== adios steps) in parallel
    """
    def __init__(self):
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.io = self.adios.DeclareIO("ioWriter")
        
        totalenergy_type = np.array([0], dtype=np.float64)
        nodal_type = np.array([3,3,3], dtype=np.float64)

        self.var_totalenergy = self.io.DefineVariable("Total_energy", totalenergy_type, [], [], [1],     adios2.ConstantDims)
        self.var_nodalinput  = self.io.DefineVariable("Nodal_input",  nodal_type,       [], [], [3,3,3], adios2.ConstantDims)
        self.var_nodaloutput = self.io.DefineVariable("Nodal_output", nodal_type,       [], [], [3,3,3], adios2.ConstantDims)
        self.var_nodalposition = self.io.DefineVariable("Nodal_position", nodal_type,   [], [], [3,3,3,3], adios2.ConstantDims)
        
        self.f = self.io.Open("ising_dataset_1.bp", adios2.Mode.Write, MPI.COMM_SELF)


    def write(self, total_energy, atomic_features, count_config, dirpath):
        # 3D array to hold the nodal input and output values
        nodal_input  = np.zeros((3,3,3))
        nodal_output = np.zeros((3,3,3))
        nodal_position = np.zeros((3,3,3,3))

        # Populate the nodal input and output arrays
        for index in range(atomic_features.shape[0]):
            x = int(atomic_features[index, 1])
            y = int(atomic_features[index, 2])
            z = int(atomic_features[index, 3])

            nodal_input[x,y,z]  = atomic_features[index, 0]
            nodal_output[x,y,z] = atomic_features[index, 4]
            nodal_position[x,y,z] = (x,y,z)

        # Now write the two 3D arrays to file
        # self.f.BeginStep()
        self.f.Put(self.var_totalenergy, np.array([total_energy]))
        self.f.Put(self.var_nodalinput,  nodal_input)
        self.f.Put(self.var_nodaloutput, nodal_output)
        self.f.Put(self.var_nodalposition, nodal_position)
        # self.f.EndStep()

    def close(self):
        self.f.Close()


class AdiosWriter_2:
    """
    This implementation creates a separate set of variables for every graph.
    For example, if there are 20k graphs, the adios file will have 20k variables each for
    the total energy, nodal input, and nodal output.
    Pros: Allows data parallelism, as each rank can writes its own graph (== adios variables) 
    separately.
    Cons: This is not a compact format. The metadata overhead could be higher (needs to be verified).
    """
    def __init__(self):
        self.adios = adios2.ADIOS(MPI.COMM_SELF)
        self.io = self.adios.DeclareIO("ioWriter")
        self.f = self.io.Open("ising_dataset_2.bp", adios2.Mode.Write, MPI.COMM_SELF)

    def write(self, total_energy, atomic_features, count_config, dirpath):
        totalenergy_type = np.array([0], dtype=np.float64)
        nodal_type = np.array([3,3,3,2], dtype=np.float64)

        self.var_totalenergy = self.io.DefineVariable("Total_energy_{}".format(count_config), totalenergy_type, [], [], [1],       adios2.ConstantDims)
        self.var_nodalparams = self.io.DefineVariable("Nodal_params_{}".format(count_config), nodal_type,       [], [], [3,3,3,2], adios2.ConstantDims)

        # 3D array to hold the nodal input and output values
        nodal_params = np.zeros((3,3,3,2))

        # Populate the nodal input and output arrays
        for index in range(atomic_features.shape[0]):
            x = int(atomic_features[index, 1])
            y = int(atomic_features[index, 2])
            z = int(atomic_features[index, 3])

            nodal_params[x,y,z,0] = atomic_features[index, 0]  # nodal input
            nodal_params[x,y,z,1] = atomic_features[index, 4]  # nodal output
        
        # self.f.BeginStep()
        total_energy_nparr = np.array([total_energy])
        self.f.Put(self.var_totalenergy, total_energy_nparr)
        self.f.Put(self.var_nodalparams, nodal_params)
        self.f.PerformPuts()
        # self.f.EndStep()
        print(total_energy_nparr)

    def close(self):
        self.f.Close()



def write_to_file(total_energy, atomic_features, count_config, dirpath):

    numpy_string_total_value = np.array2string(total_energy)

    filetxt = numpy_string_total_value

    for index in range(atomic_features.shape[0]):
        numpy_row = atomic_features[index, :]
        numpy_string_row = np.array2string(
            numpy_row, precision=2, separator="\t", suppress_small=True
        )
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

        filename = os.path.join(dirpath, "output" + str(count_config) + ".txt")
    
    with open(filename, "w") as f:
        f.write(filetxt)


# 3D Ising model
def E_dimensionless(config, L, spin_function, scale_spin):
    total_energy = 0
    spin = np.zeros_like(config)

    if scale_spin:
        random_scaling = np.random.random((L, L, L))
        config = np.multiply(config, random_scaling)

    for z in range(L):
        for y in range(L):
            for x in range(L):
                spin[x, y, z] = spin_function(config[x, y, z])

    count_pos = 0
    number_nodes = L ** 3
    positions = np.zeros((number_nodes, 3))
    atomic_features = np.zeros((number_nodes, 5))
    for z in range(L):
        for y in range(L):
            for x in range(L):
                positions[count_pos, 0] = x
                positions[count_pos, 1] = y
                positions[count_pos, 2] = z

                S = spin[x, y, z]
                nb = (
                    spin[(x + 1) % L, y, z]
                    + spin[x, (y + 1) % L, z]
                    + spin[(x - 1) % L, y, z]
                    + spin[x, (y - 1) % L, z]
                    + spin[x, y, z]
                    + spin[x, y, (z + 1) % L]
                    + spin[x, y, (z - 1) % L]
                )
                total_energy += -nb * S

                atomic_features[count_pos, 0] = config[x, y, z]
                atomic_features[count_pos, 1:4] = positions[count_pos, :]
                atomic_features[count_pos, 4] = spin[x, y, z]

                count_pos = count_pos + 1

    total_energy = total_energy / 6

    return total_energy, atomic_features


def create_dataset(
    L, histogram_cutoff, dirpath, spin_function=lambda x: x, scale_spin=False
):

    count_config = 0

    # Initialize adios writer
    adios_writer = AdiosWriter_2()

    for num_downs in tqdm(range(0, L ** 3)):

        primal_configuration = np.ones((L ** 3,))
        for down in range(0, num_downs):
            primal_configuration[down] = -1.0

        # If the current composition has a total number of possible configurations above
        # the hard cutoff threshold, a random configurational subset is picked
        if scipy.special.binom(L ** 3, num_downs) > histogram_cutoff:
            for num_config in range(0, histogram_cutoff):
                config = np.random.permutation(primal_configuration)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(
                    config, L, spin_function, scale_spin
                )

                # Leave the original I/O on for now
                write_to_file(total_energy, atomic_features, count_config, dirpath)
                adios_writer.write(total_energy, atomic_features, count_config, dirpath)

                count_config = count_config + 1

        # If the current composition has a total number of possible configurations smaller
        # than the hard cutoff, then all possible permutations are generated
        else:
            for config in multiset_permutations(primal_configuration):
                config = np.array(config)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(
                    config, L, spin_function, scale_spin
                )

                # Leave the original I/O on for now
                write_to_file(total_energy, atomic_features, count_config, dirpath)
                adios_writer.write(total_energy, atomic_features, count_config, dirpath)

                count_config = count_config + 1

    var = adios_writer.io.DefineVariable("num_config", np.array([0], dtype=np.int32), [], [], [1], adios2.ConstantDims)
    adios_writer.f.Put(var, np.array([count_config]))
    adios_writer.close()



if __name__ == "__main__":

    dirpath = os.path.join(os.path.dirname(__file__), "../../dataset/ising_model")
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

    number_atoms_per_dimension = 3
    configurational_histogram_cutoff = 1000
    # configurational_histogram_cutoff = 1

    # Use sine function as non-linear extension of Ising model
    # Use randomized scaling of the spin magnitudes
    spin_func = lambda x: math.sin(math.pi * x / 2)
    create_dataset(
        number_atoms_per_dimension,
        configurational_histogram_cutoff,
        dirpath,
        spin_function=spin_func,
        scale_spin=True,
    )
