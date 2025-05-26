import pandas as pd

# Load data
basin_df = pd.read_csv("D:/torchhydro/data/basin_531.csv")  # has 'id'
metadata_df = pd.read_csv("D:/data/waterism/datasets-origin/camels\camels_us/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata/regions.csv")  # has 'GAGE_ID' and 'HUC_02'

# Standardize ID format: zero-pad both columns to 8 digits as strings
basin_df["id"] = basin_df["id"].astype(str).str.zfill(8)
metadata_df["id"] = metadata_df["id"].astype(str).str.zfill(8)

# Merge on standardized ID
merged_df = basin_df.merge(metadata_df[["id", "HUC_02"]], left_on="id", right_on="id", how="left")

# Check for missing
missing = merged_df[merged_df["HUC_02"].isna()]
if not missing.empty:
    print("⚠️ Warning: Some basins are missing HUC region info!")
    print(missing)

# Save region-wise basin lists
for region_id in range(1, 19):
    region_basins = merged_df[merged_df["HUC_02"] == region_id]
    region_basins[["id"]].to_csv(f"gage_ids_region{region_id}.csv", index=False)
