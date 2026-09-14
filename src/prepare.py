from Load_data import load_raw
import pandas as pd

#Load the raw data
raw = load_raw()

#Get Plant 1 data
gen1 = raw["gen1"].copy()
sensor1 = raw["sensor1"].copy()

#Raw data rows and columns
print("Plant 1 Generation data shape:", gen1.shape)
print("Plant 1 Sensor data shape:", sensor1.shape)

#Inspecting raw DATE_TIME strings before conversion
print("\nRaw Gen1 DATE_TIME:")
print(gen1["DATE_TIME"].head())

print("\nRaw Sensor1 DATE_TIME:")
print(sensor1["DATE_TIME"].head())

#Converting DATE_TIME columns to datetime format
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

#Remove merge column
merged = merged.drop(columns="_merge")

#Select required columns
hourly = merged[["DATE_TIME","AC_POWER","DC_POWER","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]].copy()

# Set DATE_TIME as index
hourly = hourly.set_index("DATE_TIME")

# Resample to hourly means
hourly = hourly.resample("1h").mean()

# Reset index
hourly = hourly.reset_index()

# Rename columns
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

# Check missing values
print("\nMissing values in each column:")
print(hourly.isna().sum())

print("\nTotal rows containing at least one missing value:",hourly.isna().any(axis=1).sum())

# Handle missing values: drop rows with no ac_power
rows_before = len(hourly)
hourly = hourly.dropna().reset_index(drop=True)
rows_after = len(hourly)

print("\nRows before handling missing values:", rows_before)
#rows with no ac_power are dropped, as they are not useful for analysis because they do not provide any information about the plant's performance.
print("Rows after handling missing values:", rows_after)

# Save hourly data
hourly.to_csv("data/plant1_hourly.csv", index=False)

print("\nHourly data saved successfully.")
