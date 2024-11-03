"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:55:24
LastEditTime: 2024-11-02 21:47:38
LastEditors: Wenyu Ouyang
Description: Test funcs for seq2seq model
FilePath: /torchhydro/tests/test_seq2seq.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import torch
from torchhydro.models.seq2seq import GeneralSeq2Seq

import logging
import os.path
import pathlib

import pandas as pd
import hydrodatasource.configs.config as hdscc
import xarray as xr
import torch.multiprocessing as mp

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.deep_hydro import train_worker
from torchhydro.trainers.trainer import train_and_evaluate

# from torchhydro.trainers.trainer import train_and_evaluate, ensemble_train_and_evaluate

logging.basicConfig(level=logging.INFO)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

gage_id = [
    "songliao_21401050",
    "songliao_21401550",
]


@pytest.fixture()
def seq2seq_config():
    project_name = os.path.join("train_with_gpm", "ex_test")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source": "HydroMean",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[0],
        model_name="Seq2Seq",
        model_hyperparam={
            "en_input_size": 17,
            "de_input_size": 18,
            "output_size": 2,
            "hidden_size": 256,
            "forecast_length": 56,
            "prec_window": 1,
            "teacher_forcing_ratio": 0.5,
        },
        model_loader={"load_way": "best"},
        gage_id=gage_id,
        # gage_id=["21400800", "21401550", "21401300", "21401900"],
        batch_size=128,
        forecast_history=240,
        forecast_length=56,
        min_time_unit="h",
        min_time_interval=3,
        var_t=[
            "precipitationCal",
            "sm_surface",
        ],
        var_c=[
            "area",  # 面积
            "ele_mt_smn",  # 海拔(空间平均)
            "slp_dg_sav",  # 地形坡度 (空间平均)
            "sgr_dk_sav",  # 河流坡度 (平均)
            "for_pc_sse",  # 森林覆盖率
            "glc_cl_smj",  # 土地覆盖类型
            "run_mm_syr",  # 陆面径流 (流域径流的空间平均值)
            "inu_pc_slt",  # 淹没范围 (长期最大)
            "cmi_ix_syr",  # 气候湿度指数
            "aet_mm_syr",  # 实际蒸散发 (年平均)
            "snw_pc_syr",  # 雪盖范围 (年平均)
            "swc_pc_syr",  # 土壤水含量
            "gwt_cm_sav",  # 地下水位深度
            "cly_pc_sav",  # 土壤中的黏土、粉砂、砂粒含量
            "dor_pc_pva",  # 调节程度
        ],
        var_out=["streamflow", "sm_surface"],
        dataset="Seq2SeqDataset",
        sampler="HydroSampler",
        scaler="DapengScaler",
        train_epoch=2,
        save_epoch=1,
        train_mode=True,
        train_period=["2016-06-01-01", "2016-08-01-01"],
        test_period=["2015-06-01-01", "2015-08-01-01"],
        valid_period=["2015-06-01-01", "2015-08-01-01"],
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [0.8, 0.2],
        },
        opt="Adam",
        lr_scheduler={
            "lr": 0.0001,
            "lr_factor": 0.9,
        },
        which_first_tensor="batch",
        rolling=False,
        long_seq_pred=False,
        calc_metrics=False,
        early_stopping=True,
        # ensemble=True,
        # ensemble_items={
        #     "batch_sizes": [256, 512],
        # },
        patience=10,
        model_type="MTL",
    )

    # 更新默认配置
    update_cfg(config_data, args)

    return config_data


def test_seq2seq(seq2seq_config):
    # world_size = len(config["training_cfgs"]["device"])
    # mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    train_and_evaluate(seq2seq_config)
    # ensemble_train_and_evaluate(config)


@pytest.fixture
def model():
    return GeneralSeq2Seq(
        en_input_size=2,
        de_input_size=3,
        output_size=2,
        hidden_size=20,
        forecast_length=5,
        prec_window=10,
        teacher_forcing_ratio=0.5,
    )


def test_forward_no_teacher_forcing(model):
    src1 = torch.randn(3, 10, 2)
    src2 = torch.randn(3, 5, 1)
    outputs = model(src1, src2)
    assert outputs.shape == (3, 6, 2)


def test_forward_with_teacher_forcing(model):
    src1 = torch.randn(3, 10, 2)
    src2 = torch.randn(3, 5, 1)
    trgs = torch.randn(3, 15, 2)
    outputs = model(src1, src2, trgs)
    assert outputs.shape == (3, 6, 2)
