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
        n = self.dataset.training_cfgs["batch_size"]
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
    times = dataset.times

    # Initialize basin_groups to map each basin to a list of indices
    basin_groups = defaultdict(list)

    # Populate basin_groups with indices for each basin
    for idx, (basin_index, date) in lookup_table.items():
        actual_basin = basins[basin_index]
        actual_time = times[date]
        basin_groups[actual_basin].append((actual_basin, actual_time))

    # number of users corredponds to the number of basins
    num_users = len(basins)
    # group basins by user

    user_basins = _user_has_which_basins(basins)

    # a lookup_table subset for each user
    user_lookup_tables = {}
    for user_id, assigned_basins in user_basins.items():
        user_lookup_table = {}
        for basin in assigned_basins:
            # Iterate over the entries in basin_groups for the current basin; doesn't matter the index start from 0 or 1
            for idx, entry in enumerate(basin_groups[basin], start=1):
                user_lookup_table[idx] = entry
        user_lookup_tables[user_id] = user_lookup_table

    return user_lookup_tables


def _user_has_which_basins(basins):
    """
    A function to decide which user has which basins
    """
    user_basins = defaultdict(list)
    for i, basin in enumerate(basins):
        # user_id = i % num_users # check why use %
        user_id = i
        user_basins[user_id].append(basin)
    return user_basins


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
    times = dataset.times  # List of time indices
    from collections import defaultdict

def fl_sample_region(dataset):
    """
    Assigns each HUC region (1–18) as a federated learning client,
    returning only (basin, date) pairs.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset containing a lookup table mapping indices to (region_index, basin_index, date).

    Returns
    -------
    user_lookup_tables : dict
        A dictionary where:
        - Keys are user IDs (clients), corresponding to HUC regions (0–17).
        - Values are lookup tables mapping basins to lists of (basin, date) tuples.
    region_users : list
        List of user IDs (region indices) that were found in the dataset.
    """

    lookup_table = dataset.lookup_table  # {index -> (region_index, basin_index, date)}
    basins = dataset.basins              # List of basin names
    times = dataset.times                # List of time indices

    # Hardcoded HUC_02 -> List of basin IDs
    region_basins = {
    "01": [
        "01013500", "01022500", "01030500", "01031500", "01047000", "01052500",
        "01054200", "01055000", "01057000", "01073000", "01078000", "01118300",
        "01121000", "01123000", "01134500", "01137500", "01139000", "01139800",
        "01142500", "01144000", "01162500", "01169000", "01170100", "01181000",
        "01187300", "01195100", "04296000"
    ],
    "02": [
        "01333000", "01350000", "01350080", "01350140", "01365000", "01411300",
        "01413500", "01414500", "01415000", "01423000", "01434025", "01435000",
        "01439500", "01440000", "01440400", "01451800", "01466500", "01484100",
        "01485500", "01486000", "01487000", "01491000", "01510000", "01516500",
        "01518862", "01532000", "01539000", "01542810", "01543000", "01543500",
        "01544500", "01545600", "01547700", "01548500", "01549500", "01550000",
        "01552000", "01552500", "01557500", "01567500", "01568000", "01580000",
        "01583500", "01586610", "01591400", "01594950", "01596500", "01605500",
        "01606500", "01613050", "01620500", "01632000", "01632900", "01634500",
        "01638480", "01639500", "01644000", "01658500", "01664000", "01666500",
        "01667500", "01669000", "01669520", "02011400", "02011460", "02013000",
        "02014000", "02015700", "02016000", "02017500", "02018000", "02027000",
        "02027500", "02028500", "02038850"
    ],
    "03": [
        "02046000", "02051000", "02051500", "02053200", "02053800", "02055100",
        "02056900", "02059500", "02064000", "02065500", "02069700", "02070000",
        "02074500", "02077200", "02081500", "02082950", "02092500", "02096846",
        "02102908", "02108000", "02111180", "02111500", "02112120", "02112360",
        "02118500", "02125000", "02128000", "02137727", "02140991", "02143000",
        "02143040", "02149000", "02152100", "02177000", "02178400", "02193340",
        "02196000", "02198100", "02202600", "02212600", "02215100", "02216180",
        "02221525", "02231000", "02231342", "02235200", "02245500", "02246000",
        "02296500", "02297155", "02297310", "02298123", "02298608", "02299950",
        "02300700", "02310947", "02312200", "02314500", "02315500", "02324400",
        "02327100", "02342933", "02349900", "02350900", "02361000", "02363000",
        "02369800", "02371500", "02372250", "02374500", "02381600", "02384540",
        "02395120", "02408540", "02415000", "02422500", "02427250", "02430085",
        "02430615", "02450250", "02464000", "02464146", "02464360", "02465493",
        "02469800", "02472000", "02472500", "02479155", "02479300", "02479560",
        "02481000", "02481510"
    ],
    "04": [
        "04015330", "04024430", "04027000", "04040500", "04043050", "04045500",
        "04056500", "04057510", "04057800", "04059500", "04063700", "04074950",
        "04105700", "04115265", "04122200", "04122500", "04124000", "04127918",
        "04127997", "04161580", "04185000", "04196800", "04197100", "04197170",
        "04213000", "04213075", "04216418", "04221000", "04224775", "04233000",
        "04256000"
    ],
    "05": [
        "03010655", "03011800", "03015500", "03021350", "03026500", "03028000",
        "03049000", "03049800", "03050000", "03066000", "03069500", "03070500",
        "03076600", "03078000", "03140000", "03144000", "03159540", "03161000",
        "03164000", "03165000", "03170000", "03173000", "03180500", "03182500",
        "03186500", "03187500", "03213700", "03237280", "03237500", "03238500",
        "03241500", "03280700", "03281100", "03281500", "03285000", "03291780",
        "03300400", "03338780", "03340800", "03346000", "03357350", "03364500",
        "03366500", "03368000", "03384450"
    ],
    "06": [
        "03439000", "03450000", "03455500", "03456500", "03460000", "03463300",
        "03471500", "03473000", "03479000", "03488000", "03498500", "03500000",
        "03500240", "03504000", "03574500", "03592718", "03604000"
    ],
    "07": [
        "05291000", "05362000", "05393500", "05399500", "05408000", "05412500",
        "05413500", "05414000", "05444000", "05454000", "05458000", "05466500",
        "05487980", "05488200", "05489000", "05495000", "05495500", "05501000",
        "05503800", "05507600", "05508805", "05514500", "05525500", "05556500",
        "05584500", "05585000", "05591550", "05592050", "05592575", "05593575",
        "05593900", "05595730", "07014500"
    ],
    "08": [
        "07290650", "07291000", "07292500", "07295000", "07359610", "07362100",
        "07362587", "07373000", "07375000", "07376000", "08013000", "08014500"
    ],
    "09": [
        "05056000", "05057000", "05057200", "05062500", "05087500", "05120500",
        "05123400", "05129115", "05131500"
    ],
    "10": [
        "06037500", "06043500", "06154410", "06188000", "06191500", "06221400",
        "06224000", "06278300", "06280300", "06289000", "06291500", "06311000",
        "06332515", "06339100", "06339500", "06344600", "06350000", "06352000",
        "06353000", "06354000", "06360500", "06404000", "06406000", "06408700",
        "06409000", "06431500", "06440200", "06441500", "06447000", "06447500",
        "06450500", "06452000", "06453600", "06464500", "06468170", "06468250",
        "06470800", "06477500", "06479215", "06479438", "06601000", "06614800",
        "06622700", "06623800", "06632400", "06746095", "06784000", "06803510",
        "06803530", "06814000", "06847900", "06853800", "06876700", "06878000",
        "06879650", "06885500", "06888500", "06889200", "06889500", "06892000",
        "06903400", "06906800", "06910800", "06911900", "06917000", "06918460",
        "06919500", "06921070", "06921200", "06934000"
    ],
    "11": [
        "07056000", "07057500", "07060710", "07066000", "07067000", "07068000",
        "07071500", "07083000", "07142300", "07145700", "07148400", "07149000",
        "07151500", "07167500", "07180500", "07184000", "07195800", "07196900",
        "07197000", "07208500", "07226500", "07261000", "07263295", "07299670",
        "07301410", "07301500", "07315200", "07315700", "07335700", "07340300",
        "07346045"
    ],
    "12": [
        "08023080", "08025500", "08029500", "08050800", "08066200", "08066300",
        "08070000", "08070200", "08079600", "08082700", "08086212", "08086290",
        "08101000", "08103900", "08104900", "08109700", "08150800", "08155200",
        "08158700", "08158810", "08164000", "08164300", "08164600", "08165300",
        "08171300", "08175000", "08176900", "08178880", "08189500", "08190000",
        "08190500", "08194200", "08195000", "08196000", "08198500", "08200000",
        "08202700"
    ],
    "13": [
        "08267500", "08269000", "08271000", "08324000", "08377900", "08378500",
        "08380500"
    ],
    "14": [
        "09034900", "09035800", "09035900", "09047700", "09065500", "09066000",
        "09066200", "09066300", "09081600", "09107000", "09210500", "09223000",
        "09306242", "09312600", "09352900", "09378170", "09378630"
    ],
    "15": [
        "09386900", "09404450", "09430500", "09430600", "09447800", "09480000",
        "09484000", "09484600", "09492400", "09494000", "09497800", "09497980",
        "09505200", "09505350", "09505800", "09508300", "09510200", "09512280",
        "09513780"
    ],
    "16": [
        "10023000", "10166430", "10172700", "10172800", "10173450", "10205030",
        "10234500", "10242000", "10244950", "10249300", "10310500", "10316500",
        "10329500", "10336645", "10336660", "10336740", "10343500", "10348850"
    ],
    "17": [
        "10396000", "12010000", "12013500", "12020000", "12025000", "12025700",
        "12035000", "12040500", "12041200", "12043000", "12048000", "12054000",
        "12056500", "12073500", "12082500", "12092000", "12095000", "12114500",
        "12115000", "12115500", "12117000", "12141300", "12143600", "12144000",
        "12145500", "12147500", "12147600", "12167000", "12175500", "12178100",
        "12186000", "12189500", "12358500", "12374250", "12375900", "12377150",
        "12381400", "12383500", "12388400", "12390700", "12411000", "12414500",
        "12447390", "12451000", "12488500", "13011500", "13011900", "13018300",
        "13023000", "13083000", "13161500", "13235000", "13240000", "13310700",
        "13313000", "13331500", "13337000", "13338500", "13340000", "13340600",
        "14020000", "14092750", "14096850", "14137000", "14138800", "14138870",
        "14138900", "14139800", "14141500", "14154500", "14158500", "14158790",
        "14166500", "14182500", "14185000", "14185900", "14187000", "14216500",
        "14222500", "14236200", "14301000", "14303200", "14305500", "14306340",
        "14306500", "14308990", "14309500", "14316700", "14325000", "14362250",
        "14400000"
    ],
    "18": [
        "10258000", "10258500", "10259000", "10259200", "10263500", "11098000",
        "11124500", "11141280", "11143000", "11148900", "11151300", "11162500",
        "11176400", "11180500", "11180960", "11224500", "11230500", "11237500",
        "11253310", "11264500", "11266500", "11274500", "11274630", "11284400",
        "11299600", "11381500", "11383500", "11451100", "11468500", "11473900",
        "11475560", "11476600", "11478500", "11480390", "11481200", "11482500",
        "11522500", "11523200", "11528700", "11532500"
    ]
}


    # Number of users corresponds to the number of regions
    # num_users = len(region_basins)

    # Initialize basin_groups to map each basin to a list of indices
    basin_groups = defaultdict(list)

    # Populate basin_groups with indices for each basin
    for idx, (basin_index, date) in lookup_table.items():
        actual_basin = basins[basin_index]
        actual_time = times[date]
        basin_groups[actual_basin].append((actual_basin, actual_time))

    # Create lookup tables for each region (user)
    user_lookup_tables = defaultdict(dict)
    user_id = 0  # This will be used to assign users (regions)
    # Iterate through the regions and assign basins to the appropriate region (user)
    for region, region_basin_list in region_basins.items():
        user_lookup_table = defaultdict(
            list
        )  # Use list to store multiple (basin, date) pairs for each basin

        # Track if any basin from user-provided basins belongs to the current region
        region_has_basins = False

        # Check for each basin in user-provided basins
        for basin in basins:
            # Ensure the basin is properly formatted (leading zeros if needed)
            basin = str(basin).zfill(
                8
            )  # Ensure basin is treated as string with leading zeros

            # Check if the basin exists in the basin_groups
            if basin in basin_groups:
                # Check if the basin belongs to the current region (from region_basin_list)
                if basin in region_basin_list:
                    region_has_basins = True  # Mark that this region has basins

                    # Add all the (basin, date) pairs to the user_lookup_table for this region
                    user_lookup_table[basin].extend(
                        basin_groups[basin]
                    )  # Extend to store all (basin, date)

        # Only add the user_lookup_table if the region has basins
        if region_has_basins:
            region_number = int(
                region.split()[-1]
            )  # This will extract the number from 'Region X'
            user_id = region_number - 1  # Since region is 1-based, user_id is region-1

            user_lookup_tables[user_id] = dict(
                user_lookup_table
            )  # Convert defaultdict to regular dict
    # Calculate the number of users (regions)
    region_users = list(user_lookup_tables.keys())
    return user_lookup_tables, region_users


data_sampler_dict = {
    "KuaiSampler": KuaiSampler,
    "BasinBatchSampler": BasinBatchSampler,
    # TODO: DistributedSampler need more test
    # "DistSampler": DistributedSampler,
}
