##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import os
import numpy as np
import pickle
import pathlib
import adios2

import torch
from torch_geometric.data import Data
from torch import tensor

# WARNING: DO NOT use collective communication calls here because only rank 0 uses this routines


def tensor_divide(x1, x2):
    return torch.from_numpy(np.divide(x1, x2, out=np.zeros_like(x1), where=x2 != 0))


class AdiosDataLoader:
    """A class used for loading raw files that contain data representing atom structures, transforms it and stores the structures as file of serialized structures.
    Most of the class methods are hidden, because from outside a caller needs only to know about
    load_raw_data method.

    Methods
    -------
    load_raw_data()
        Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    """

    def __init__(self, filename):
        """
        config:
          shows the dataset path the target variables information, e.g, location and dimension, in data file
        """
        self.dataset_list = []
        self.serial_data_name_list = []
        # self.node_feature_name = config["node_features"]["name"]
        # self.node_feature_dim = config["node_features"]["dim"]
        # self.node_feature_col = config["node_features"]["column_index"]
        # self.graph_feature_name = config["graph_features"]["name"]
        # self.graph_feature_dim = config["graph_features"]["dim"]
        # self.graph_feature_col = config["graph_features"]["column_index"]
        # self.raw_dataset_name = config["name"]
        # self.data_format = config["format"]
        # self.path_dictionary = config["path"]

        self.filename = filename

    def load_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        dataset = []
        with adios2.open(self.filename, 'r') as f:
            num_config = f.read('num_config')[0]
            for i in range(num_config):
                data_object = self.transform_variable_to_data_object_base(f, i)
                dataset.append(data_object)
        
        return dataset

    def transform_variable_to_data_object_base(self, f, index: int):
        """Transforms lines of strings read from the raw data file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        data_object = Data()

        g_feature_varname = 'Total_energy_%d'%index
        node_input_feature_varname = 'Nodal_input_%d'%index
        node_output_feature_varname = 'Nodal_output_%d'%index
        node_position_varname = 'Nodal_position_%d'%index

        g_feature = f.read('Total_energy_%d'%index)
        nodal_input = f.read('Nodal_input_%d'%index)
        nodal_output = f.read('Nodal_output_%d'%index)
        nodal_position = f.read('Nodal_position_%d'%index)
        d0, d1, d2, d3 = nodal_position.shape 
        assert (d3 == 3)
        
        nodal_feature = np.vstack((nodal_input.ravel(), nodal_output.ravel())).T
        nodal_position = np.reshape(nodal_position, (d0*d1*d2, d3))

        data_object.y = tensor(g_feature)
        data_object.x = tensor(nodal_feature)
        data_object.pos = tensor(nodal_position)

        return data_object

if __name__ == "__main__":
    loader = AdiosDataLoader('ising_dataset_2.bp')
    dataset = loader.load_data()
