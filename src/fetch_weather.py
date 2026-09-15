import requests
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive-api.open-meteo.com/v1/archive"
parameters = {
    "latitude": 14.82,
    "longitude": 78.28,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata"
}

# Get weather data from Open Meteo and prepare it for comparison with plant data
response = requests.get(url, params=parameters)
print("Status code:", response.status_code)
data = response.json()
hourly_data = data["hourly"]
weather = pd.DataFrame(hourly_data)
weather = weather.rename(columns={
    "time": "datetime",
    "shortwave_radiation": "sw_radiation",
    "temperature_2m": "temp_2m",
    "cloud_cover": "cloud_cover"
})
weather["sw_radiation"] = weather["sw_radiation"] / 1000
print(weather.head())
print("Weather data shape:", weather.shape)

# Save the weather data
weather.to_csv("data/plant1_openmeteo.csv", index=False)
print("Open-Meteo data saved successfully.")

# Load plant hourly data
plant = pd.read_csv("data/plant1_hourly.csv")
plant["datetime"] = pd.to_datetime(plant["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])

# Combine plant and weather data and plot irradiation comparison for three days
comparison = pd.merge(plant,weather,on="datetime",how="inner")
three_days = comparison[(comparison["datetime"] >= "2020-05-15") &(comparison["datetime"] < "2020-05-18")]
plt.plot(three_days["datetime"],three_days["irradiation"],label="Plant Sensor")
plt.plot(three_days["datetime"],three_days["sw_radiation"],label="Open-Meteo")
plt.xlabel("Datetime")
plt.ylabel("Radiation (kW/m2)")
plt.title("Plant Sensor vs Open-Meteo Radiation")
plt.legend()
plt.savefig("results/figures/radiation_comparison_3days.png")
plt.show()

# Peak hour of each curve for the three days
three_days = three_days.copy()
three_days["date"] = three_days["datetime"].dt.date
peaks = []
for day, group in three_days.groupby("date"):
    sensor_peak = group.loc[group["irradiation"].idxmax(), "datetime"].hour
    meteo_peak = group.loc[group["sw_radiation"].idxmax(), "datetime"].hour
    print(f"{day}: sensor peak {sensor_peak}:00, Open-Meteo peak {meteo_peak}:00")
    peaks.append(f"{day}: {sensor_peak}:00 vs {meteo_peak}:00")

# Table 1 - add the Open-Meteo rows to the part saved by prepare.py
table1 = pd.read_csv("results/table/table1_data_preparation.csv")
table1 = table1[~table1["Item"].isin(["Open-Meteo rows downloaded",
                                      "Peak hour, sensor irradiation vs. Open-Meteo radiation (three days)"])]
table1.loc[len(table1)] = ["Open-Meteo rows downloaded", len(weather)]
table1.loc[len(table1)] = ["Peak hour, sensor irradiation vs. Open-Meteo radiation (three days)",
                           "; ".join(peaks)]
table1.to_csv("results/table/table1_data_preparation.csv", index=False)
print("\nTable 1 - Data preparation")
print(table1.to_string(index=False))