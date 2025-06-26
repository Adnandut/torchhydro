"""
Author: Wenyu Ouyang
Date: 2023-09-24 14:28:48
LastEditTime: 2023-12-18 09:11:55
LastEditors: Wenyu Ouyang
Description: A test for federated learning
FilePath: \torchhydro\tests\test_federated_learning.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import os
import os
import numpy as np


@pytest.fixture()
def config():
    project_name = os.path.join("test_camels", "testglobal14")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "camels_us",
            "source_path": os.path.join(
                SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
            ),
        },
        ctx=[-1],
        model_type="FedLearn",
        model_name="CpuLSTM",
        model_hyperparam={
            "n_input_features": 31,
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
            "01052500",
            "01054200",
            "14216500",
            "14222500",
            "14236200",
            "14301000",
            "14303200",
            "14305500",
        ],
        forecast_history=0,
        forecast_length=30,
        var_t=["prcp", "srad", "tmax", "tmin", "vp"],
        var_c=[
            "elev_mean",
            "slope_mean",
            "area_gages2",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "gvf_max",
            "gvf_diff",
            "soil_depth_pelletier",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "sand_frac",
            "silt_frac",
            "clay_frac",
            "carbonate_rocks_frac",
            "geol_permeability",
            "p_mean",
            "pet_mean",
            "aridity",
            "frac_snow",
            "high_prec_freq",
            "high_prec_dur",
            "low_prec_freq",
            "low_prec_dur",
        ],
        var_out=["streamflow"],
        dataset="StreamflowDataset",
        scaler="StandardScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": ["prcp"],
            "pbm_norm": False,
        },
        train_epoch=10,
        save_epoch=1,
        start_epoch=1,
        fl_sample="basin",
        fl_frac=1,
        fl_num_users=13,
        fl_local_bs=128,
        fl_local_ep=4,
        fedprox_mu=0.001,
        early_stopping=True,
        patience=1,
        target_as_input=False,
        ensemble=False,
        pretrain_path="D:\pretrain_model\model_Ep30.pth",
        dropout=0.2,
        model_loader={"load_way": "best", "test_epoch": "best_model.pth"},
        train_period=["2007-10-01", "2008-10-01"],
        valid_period=["2008-10-01", "2009-10-01"],
        test_period=["2009-10-01", "2010-10-01"],
        loss_func="RMSESum",
        opt="Adam",
        rs=1111,
        # key is epoch, start from 1
        opt_param={
            "lr": 0.001,
        },
        lr_scheduler={3: 0.01, 5: 0.001},
        which_first_tensor="sequence",
    )
    update_cfg(config_data, args)
    return config_data


def test_train_evaluate(config):
    train_and_evaluate(config)
    print("federalted learning process is finished.")
