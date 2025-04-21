"""
Author: Wenyu Ouyang
Date: 2023-09-25 08:21:27
LastEditTime: 2024-11-04 18:16:08
LastEditors: Wenyu Ouyang
Description: Some sampling class or functions
FilePath: \torchhydro\torchhydro\datasets\sampler.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from collections import defaultdict
import numpy as np
from torch.utils.data import RandomSampler, Sampler
from torchhydro.datasets.data_sets import BaseDataset
from typing import Iterator, Optional
import torch


class KuaiSampler(RandomSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        warmup_length,
        rho_horizon,
        ngrid,
        nt,
    ):
        """a sampler from Kuai Fang's paper: https://doi.org/10.1002/2017GL075619
           He used a random pick-up that we don't need to iterate all samples.
           Then, we can train model more quickly

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            just a object of dataset class inherited from torch.utils.data.Dataset
        batch_size : int
            we need batch_size to calculate the number of samples in an epoch
        warmup_length : int
            warmup length, typically for physical hydrological models
        rho_horizon : int
            sequence length of a mini-batch, for encoder-decoder models, rho+horizon, for decoder-only models, horizon
        ngrid : int
            number of basins
        nt : int
            number of all periods
        """
        while batch_size * rho_horizon >= ngrid * nt:
            # try to use a smaller batch_size to make the model runnable
            batch_size = int(batch_size / 10)
        batch_size = max(batch_size, 1)
        # 99% chance that all periods' data are used in an epoch
        n_iter_ep = int(
            np.ceil(
                np.log(0.01)
                / np.log(1 - batch_size * rho_horizon / ngrid / (nt - warmup_length))
            )
        )
        assert n_iter_ep >= 1
        # __len__ means the number of all samples, then, the number of loops in an epoch is __len__()/batch_size = n_iter_ep
        # hence we return n_iter_ep * batch_size
        num_samples = n_iter_ep * batch_size
        super(KuaiSampler, self).__init__(dataset, num_samples=num_samples)


class BasinBatchSampler(Sampler[int]):
    """
    A custom sampler for hydrological modeling that iterates over a dataset in
    a way tailored for batches of hydrological data. It ensures that each batch
    contains data from a single randomly selected 'basin' out of several basins,
    with batches constructed to respect the specified batch size and the unique
    characteristics of hydrological datasets.
    TODO: made by Xinzhuo Wu, maybe need to be tested more

    Parameters
    ----------
    dataset : BaseDataset
        The dataset to sample from, expected to have a `data_cfgs` attribute.
    num_samples : Optional[int], default=None
        The total number of samples to draw (optional).
    generator : Optional[torch.Generator]
        A PyTorch Generator object for random number generation (optional).

    The sampler divides the dataset by the number of basins, then iterates through
    each basin's range in shuffled order, ensuring non-overlapping, basin-specific
    batches suitable for models that predict hydrological outcomes.
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.dataset = dataset
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        n = self.dataset.data_cfgs["batch_size"]
        basin_number = len(self.dataset.data_cfgs["object_ids"])
        basin_range = len(self.dataset) // basin_number
        if n > basin_range:
            raise ValueError(
                f"batch_size should equal or less than basin_range={basin_range} "
            )

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # basin_list = torch.randperm(basin_number)
        # for select_basin in basin_list:
        #     x = torch.randperm(basin_range)
        #     for i in range(0, basin_range, n):
        #         yield from (x[i : i + n] + basin_range * select_basin.item()).tolist()
        x = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, n):
            yield from (x[i : i + n]).tolist()

    def __len__(self) -> int:
        return self.num_samples


def fl_sample_basin(dataset: BaseDataset):
    """
    Sample one basin data as a client from a dataset for federated learning

    Parameters
    ----------
    dataset
        dataset

    Returns
    -------
        dict of image index
    """
    lookup_table = dataset.lookup_table
    basins = dataset.basins

    # Initialize basin_groups to map each basin to a list of indices
    basin_groups = defaultdict(list)

    # Populate basin_groups with indices for each basin
    for idx, (basin_index, date) in lookup_table.items():
        actual_basin = basins[basin_index]
        basin_groups[actual_basin].append((actual_basin,date))

    #number of users corredponds to the number of basins
    num_users = len(basins)
    # group basins by user

    user_basins = defaultdict(list)
    for i, basin in enumerate(basins):
        user_id = i % num_users
        user_basins[user_id].append(basin)

    # a lookup_table subset for each user
    user_lookup_tables = {}
    for user_id, assigned_basins in user_basins.items():
        user_lookup_table = {}
        for basin in assigned_basins:
            # Iterate over the entries in basin_groups for the current basin
            for idx, entry in enumerate(basin_groups[basin], start=1):
                user_lookup_table[idx] = entry
        user_lookup_tables[user_id] = user_lookup_table

    return user_lookup_tables


def fl_sample_region(dataset: BaseDataset):
    """
    Assigns each region as a federated learning client, returning only (basin, date) pairs.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing a lookup table mapping indices to (region, basin, date).

    Returns
    -------
    user_lookup_tables : dict
        A dictionary where:
        - Keys are user IDs (clients).
        - Values are lookup tables mapping indices to (basin, date) tuples.
    """
    lookup_table = dataset.lookup_table  # {index -> (region_index, basin_index, date)}
    basins = dataset.basins  # List of basin names
    import pandas as pd

    # Define the region mapping
    region_mapping = {
        "01": "Region 1",
        "02": "Region 2",
        "03": "Region 3",
        "04": "Region 4",
        "05": "Region 5",
        "06": "Region 6",
        "07": "Region 7",
        "08": "Region 8",
        "09": "Region 9",
        "10": "Region 10",
        "11": "Region 11",
        "12": "Region 12",
        "13": "Region 13",
        "14": "Region 14",
        "15": "Region 15",
        "16": "Region 16",
        "17": "Region 17",
        "18": "Region 18",
        # Add all other huc_id or basin-to-region mappings here
    }

    # Read the file that contains huc_id and basin info
    def read_and_classify_basins(file_path):
    # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, dtype={'HUC_02': str, 'GAGE_ID': str})  # Ensure 'HUC_02' and 'GAGE_ID' are strings
        
        # Dictionary to store basins by region
        region_basins = {region: [] for region in region_mapping.values()}
        
        # Iterate over the rows and classify basins based on the region
        for index, row in df.iterrows():
            # Convert HUC_02 to string and pad with leading zero if necessary
            huc_id = str(row['HUC_02']).zfill(2)  # Ensures HUC_ID is treated as a string
            basin = str(row['GAGE_ID']).zfill(8)  # Basin ID as a string, leading zeros preserved
            if huc_id in region_mapping:
                region = region_mapping[huc_id]
                region_basins[region].append(basin)
            else:
                print(f"Warning: HUC_ID {huc_id} not found in region_mapping")

        return region_basins
    # Example usage
    file_path = "D:/data/waterism/datasets-origin/camels/camels_us/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata/regions.csv"  # Replace with your file path
    region_basins = read_and_classify_basins(file_path)

    class Dataset:
        def __init__(self):
            self.regions = {}

    # Create a Dataset object and store the classified basins by region
    dataset = Dataset()
    dataset.regions = region_basins

    # Number of users corresponds to the number of regions
    # num_users = len(region_basins)
    
     # Initialize basin_groups to map each basin to a list of indices
    basin_groups = defaultdict(list)

    # Populate basin_groups with indices for each basin
    for idx, (basin_index, date) in lookup_table.items():
        actual_basin = basins[basin_index]
        basin_groups[actual_basin].append((actual_basin, date))

    # Create lookup tables for each region (user)
    user_lookup_tables = defaultdict(dict)
    user_id = 0  # This will be used to assign users (regions)
 # Iterate through the regions and assign basins to the appropriate region (user)
    for region, region_basin_list in region_basins.items():
        user_lookup_table = defaultdict(list)  # Use list to store multiple (basin, date) pairs for each basin

        # Track if any basin from user-provided basins belongs to the current region
        region_has_basins = False

        # Check for each basin in user-provided basins
        for basin in basins:
            # Ensure the basin is properly formatted (leading zeros if needed)
            basin = str(basin).zfill(8)  # Ensure basin is treated as string with leading zeros

            # Check if the basin exists in the basin_groups
            if basin in basin_groups:
                # Check if the basin belongs to the current region (from region_basin_list)
                if basin in region_basin_list:
                    region_has_basins = True  # Mark that this region has basins

                    # Add all the (basin, date) pairs to the user_lookup_table for this region
                    user_lookup_table[basin].extend(basin_groups[basin])  # Extend to store all (basin, date)

        # Only add the user_lookup_table if the region has basins
        if region_has_basins:
            region_number = int(region.split()[-1])  # This will extract the number from 'Region X'
            user_id = region_number - 1  # Since region is 1-based, user_id is region-1

            user_lookup_tables[user_id] = dict(user_lookup_table)  # Convert defaultdict to regular dict
    # Calculate the number of users (regions)
    region_users = list(user_lookup_tables.keys())
    return user_lookup_tables, region_users
data_sampler_dict = {
    "KuaiSampler": KuaiSampler,
    "BasinBatchSampler": BasinBatchSampler,
    # TODO: DistributedSampler need more test
    # "DistSampler": DistributedSampler,
}
