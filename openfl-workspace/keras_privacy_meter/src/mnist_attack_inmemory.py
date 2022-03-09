# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from openfl.federated import TensorFlowDataLoader

from .mnist_utils import load_mnist_shard

import sys
sys.path.insert(0, '~/miniconda3/envs/gpuenv/lib/python3.8/site-packages/openfl-workspace/ml_privacy_meter')
# sys.path.insert(0, '/home/aspaul/miniconda3/envs/openfl-mlprivmeter/lib/python3.7/site-packages/openfl-workspace/ml_privacy_meter')
import ml_privacy_meter

import numpy as np 

class MNISTAttackInMemory(TensorFlowDataLoader):
    """TensorFlow Data Loader for MNIST Dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initialize.

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and
        #  what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of
        # collaborator list.

        input_shape, num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(
            shard_num=int(data_path), **kwargs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        # TODO: for whitebox attack labels cannot be one hot encoded currently
        y_train_labels = np.argmax(self.y_train, axis=1)
        y_valid_labels = np.argmax(self.y_valid, axis=1)

        self.num_classes = num_classes

        num_datapoints = 5000
        x_target_train, y_target_train = self.X_train[:num_datapoints], y_train_labels[:num_datapoints]

        # population data (training data is a subset of this here)
        x_population = np.concatenate((X_train, X_valid))
        y_population = np.concatenate((y_train_labels, y_valid_labels))

        self.attack_data_handler = ml_privacy_meter.utils.attack_data.AttackData(
                                       x_population=x_population,
                                       y_population=y_population,
                                       x_target_train=x_target_train,
                                       y_target_train=y_target_train,
                                       batch_size=100,
                                       attack_percentage=10, input_shape=input_shape,
                                       normalization=True)
