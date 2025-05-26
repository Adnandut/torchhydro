import os
import pandas as pd

# Your function as is
from run_camellstm_exp import run_normal_dl  # change this to your actual script filename

# Path to the file containing all basin IDs
all_basins_file = "D:/torchhydro/data/basin_531.csv"

# Read the basin IDs
basin_ids = pd.read_csv(all_basins_file, dtype={0: str}).iloc[:, 0].values

# Create a folder to store individual basin files (optional)
os.makedirs("D:/torchhydro/data/single_basins", exist_ok=True)

# Loop over each basin
for basin_id in basin_ids:
    # Create a temporary file containing only this basin
    single_basin_file = f"D:/torchhydro/data/single_basins/{basin_id}.csv"
    with open(single_basin_file, "w") as f:
        f.write("gauge_id\n")
        f.write(f"{basin_id}\n")
    
    # Define experiment name per basin (optional, for clean logging)
    experiment_name = os.path.join("ndl", f"exp_basin_{basin_id}")

    # Run the training and evaluation
    print(f"Running for basin: {basin_id}")
    run_normal_dl(
        project_name=experiment_name,
        gage_id_file=single_basin_file
    )

print("✅ All basins processed!")
