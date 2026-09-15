from Load_data import load_raw
import pandas as pd

#Loading,inspecting and preparing raw data for Plant 1
raw = load_raw()
gen1 = raw["gen1"].copy()
sensor1 = raw["sensor1"].copy()
print("Plant 1 Generation data shape:", gen1.shape)
print("Plant 1 Sensor data shape:", sensor1.shape)
print("\nRaw Gen1 DATE_TIME:")
print(gen1["DATE_TIME"].head())
print("\nRaw Sensor1 DATE_TIME:")
print(sensor1["DATE_TIME"].head())
gen1["DATE_TIME"] = pd.to_datetime(gen1["DATE_TIME"],dayfirst=True,format="mixed")
sensor1["DATE_TIME"] = pd.to_datetime(sensor1["DATE_TIME"])

#Printing first and last timestamps after conversion for both datasets
print("\nGeneration data Timestamps: After conversion")
print(gen1["DATE_TIME"].min())
print(gen1["DATE_TIME"].max())
print("\nSensor data Timestamps: After conversion")
print(sensor1["DATE_TIME"].min())
print(sensor1["DATE_TIME"].max())

#Sum AC and DC power from all Plant 1 inverters
gen1_power_level = (gen1.groupby("DATE_TIME")[["AC_POWER", "DC_POWER"]].sum().reset_index())

#Merge plant-level power with sensor data
merged = pd.merge(gen1_power_level,sensor1,on="DATE_TIME",how="outer",indicator=True)

#Count and print timestamps
counts = merged["_merge"].value_counts()
print("\nTimestamps in both files:", counts.get("both", 0))
print("\nTimestamps only in gen1_power_level:",counts.get("left_only", 0))
print("\nTimestamps only in sensor1:",counts.get("right_only", 0))
merged = merged.drop(columns="_merge")

#Resample to hourly data
hourly = merged[["DATE_TIME","AC_POWER","DC_POWER","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]].copy()
hourly = hourly.set_index("DATE_TIME")
hourly = hourly.resample("1h").mean()
hourly = hourly.reset_index()
hourly = hourly.rename(columns={
    "DATE_TIME": "datetime",
    "AC_POWER": "ac_power",
    "DC_POWER": "dc_power",
    "AMBIENT_TEMPERATURE": "ambient_temp",
    "MODULE_TEMPERATURE": "module_temp",
    "IRRADIATION": "irradiation"
})

# Check number of hourly rows
print("\nNumber of hourly rows:", len(hourly))

# Handling missing values
print("\nMissing values in each column:")
print(hourly.isna().sum())
missing_rows = hourly.isna().any(axis=1).sum()
print("\nTotal rows containing at least one missing value:", missing_rows)
rows_before = len(hourly)
hourly = hourly.dropna().reset_index(drop=True)
rows_after = len(hourly)
print("\nRows before handling missing values:", rows_before)
print("Rows after handling missing values:", rows_after)
hourly.to_csv("data/plant1_hourly.csv", index=False)

# Table 1 - data preparation (Open-Meteo rows are added by fetch_weather.py)
table1 = pd.DataFrame({
    "Item": [
        "Raw generation rows (Plant 1)",
        "Raw sensor rows (Plant 1)",
        "Timestamps in only one file",
        "Hourly rows after resampling",
        "Hourly rows with missing values",
    ],
    "Value": [
        len(gen1),
        len(sensor1),
        counts.get("left_only", 0) + counts.get("right_only", 0),
        rows_before,
        f"{missing_rows} (dropped)",
    ],
})
table1.to_csv("results/table/table1_data_preparation.csv", index=False)
print("\nTable 1 - Data preparation (Task 1 part)")
print(table1.to_string(index=False))

