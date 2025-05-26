import os
from experiments.FL_region_531 import run_normal_dl

# Path where region-wise gage files are stored
gage_folder = "D:/torchhydro/data/gage_region_files"  # your folder with gage_ids_regionX.csv
results_root = "D:/torchhydro/results/FLRegion531"    # root output folder (can be anywhere)

# Make sure results directory exists
os.makedirs(results_root, exist_ok=True)

# Loop over all 18 regions
for region_id in range(1, 19):
    gage_file = os.path.join(gage_folder, f"gage_ids_region{region_id}.csv")
    
    # Create a unique subfolder for this run
    project_name = os.path.join(results_root, f"region_{region_id}")
    os.makedirs(project_name, exist_ok=True)

    print(f"\n=== Running Region {region_id} ===")
    print(f"Gage File: {gage_file}")
    print(f"Saving results in: {project_name}")

    run_normal_dl(
        project_name=project_name,
        gage_id_file=gage_file
    )