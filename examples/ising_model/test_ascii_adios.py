"""
Test the original I/O that uses separate ascii files for each graph/sample 
versus its adios implementation.

This assumes that the adios file has the following pattern of variables:
    Total_energy_{}, Nodal_params_{}, where {} is the graph id or count.

The last dimension of nodal_params represents the nodal input and output.
    nodal_params[:,:,:,0] is the nodal input, whereas nodal_params[:,:,:,1]
    is the nodal output.

This currently works for 3D graphs/samples only.
"""

import numpy as np
from mpi4py import MPI
import math
import adios2
from pathlib import Path


# Global filenames
ascii_dirpath = Path("../../dataset/ising_model")
adios_file = "./ising_dataset_2.bp"


def compare_files(ascii_fpath, adios_f):
    fname = ascii_fpath.name

    # Get the id from a filename like "output99.txt"
    graph_id = int(fname.split(".txt")[0].split("output")[1])

    # Read the ascii file in
    ascii_lines = []
    with open(fpath) as f:
        ascii_lines = f.readlines()
    
    # The first line is the total energy
    total_energy = float(ascii_lines[0].strip())
    adios_total_energy = adios_f.read("Total_energy_{}".format(graph_id))
    assert math.isclose(total_energy, adios_total_energy[0], rel_tol=1e-3), \
        "Total energy mismatch in {}. Ascii file has {}, adios file has {}" \
        "".format(fname, total_energy, adios_total_energy[0])

    # Now read all the 3^3 = 27 vertices of the graph
    adios_nodalparams = adios_f.read('Nodal_params_{}'.format(graph_id))
    for line in ascii_lines[1:]:
        tokens = line.split()
        nodal_input = float(tokens[0])
        nodal_output = float(tokens[4])
        x = int(float(tokens[1]))
        y = int(float(tokens[2]))
        z = int(float(tokens[3]))

        # Need to round it as the ascii file only stores two levels of precision
        adios_nodalinput  = round(adios_nodalparams[x,y,z,0], 2)
        adios_nodaloutput = round(adios_nodalparams[x,y,z,1], 2)

        assert nodal_input == adios_nodalinput, \
            "Nodal input mismatch in {}. " \
            "ascii file has {}, adios file has {}" \
            "".format(fname, nodal_input, adios_nodalinput)

        assert nodal_output == adios_nodaloutput, \
            "Nodal output mismatch in {}. " \
            "ascii file has {}, adios file has {}" \
            "".format(fname, nodal_output, adios_nodaloutput)

    print("{} verified".format(fname))


#----------------------------------------------------------------------------#
#                                 MAIN
#----------------------------------------------------------------------------#
ascii_files   = list(ascii_dirpath.glob("*.txt"))
with adios2.open(adios_file, "r") as f:
    for fpath in ascii_files:
        compare_files(fpath, f)
