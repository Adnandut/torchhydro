"""
Author: Wenyu Ouyang
Date: 2022-09-09 14:47:42
LastEditTime: 2024-11-11 18:33:10
LastEditors: Wenyu Ouyang
Description: a script to run experiments for LSTM - CAMELS
FilePath: \torchhydro\experiments\run_camelslstm_experiments.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

import torch

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


VAR_C_CHOSEN_FROM_CAMELS_US = [
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
]
VAR_T_CHOSEN_FROM_DAYMET = [
    "prcp",
    "srad",
    "tmax",
    "tmin",
    "vp",
]


def run_normal_dl(
    project_name,
    gage_id_file,
    var_c=VAR_C_CHOSEN_FROM_CAMELS_US,
    var_t=VAR_T_CHOSEN_FROM_DAYMET,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    if train_period is None:
        train_period = ["1999-10-01", "2008-09-30"]
    if valid_period is None:
        valid_period = ["1980-10-01", "1989-09-30"]
    if test_period is None:
        test_period = ["1989-10-01", "1999-09-30"]
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
        model_name="CpuLSTM",
        model_type="FedLearn",
        model_hyperparam={
            "n_input_features": len(var_c) + len(var_t),
            "n_output_features": 1,
            "n_hidden_states": 256,
        },
        loss_func="RMSESum",
        dataset="StreamflowDataset",
        scaler="StandardScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": ["prcp"],
            "pbm_norm": False,
        },
        fl_local_bs=64,
        fl_local_ep=10,
        fl_sample="basin",
        fl_frac=0.5,
        forecast_history=0,
        forecast_length=100,
        early_stopping=True,
        patience=5,
        dropout=0.4,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        var_t_type=["daymet"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt="Adam",
        rs=1234,
        train_epoch=2,
        save_epoch=1,
        model_loader={
            "load_way": "best",
            "test_epoch": "best_model.pth",
        },
        calc_metrics=True,
        opt_param={
            "lr": 0.001,
        },
        lr_scheduler={1: 0.1, 5: 0.05},
        gage_id_file=gage_id_file,
        which_first_tensor="sequence",
        metrics=["NSE", "KGE", "RMSE", "Corr", "FHV", "FLV"],
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)


# print("All processes are finished!")


# the gage_id.txt file is set by the user, it must be the format like:
# GAUGE_ID
# 01013500
# 01022500
# ......
# Then it can be read by pd.read_csv(gage_id_file, dtype={0: str}).iloc[:, 0].values to get the gage_id list
run_normal_dl(os.path.join("FL1", "expkratzert"), "D:/torchhydro/data/gage_id_test.csv")
