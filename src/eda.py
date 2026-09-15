import pandas as pd
import matplotlib.pyplot as plt

#load hourly data
hourly = pd.read_csv("data/plant1_hourly.csv")

#AC power vs irradiation
plt.scatter(hourly["irradiation"], hourly["ac_power"])
plt.xlabel("Irradiation")
plt.ylabel("AC Power")
plt.title("AC Power vs Irradiation")
plt.grid(True)
plt.savefig("results/figures/ac_power_vs_irradiation.png")
plt.show()

#Module temperature vs ambient temperature
plt.scatter(hourly["ambient_temp"],hourly["module_temp"],c=hourly["irradiation"],alpha=0.6)
plt.xlabel("Ambient Temperature")
plt.ylabel("Module Temperature")
plt.title("Module Temperature vs Ambient Temperature")
plt.colorbar(label="Irradiation")
plt.grid(True)
plt.savefig("results/figures/module_temp_vs_ambient_temp.png")
plt.show()

#AC power vs DC power
plt.scatter(hourly["dc_power"], hourly["ac_power"])
plt.xlabel("DC Power")
plt.ylabel("AC Power")
plt.title("AC Power vs DC Power")
plt.grid(True)
plt.savefig("results/figures/ac_power_vs_dc_power.png")
plt.show()

#calculate AC/DC ratio
data = hourly[hourly["dc_power"] > 0].copy()
ratio = data["ac_power"] / data["dc_power"]
print("Average AC/DC ratio:", ratio.mean())

#Average AC power for each hour
hourly["hour"] = pd.to_datetime(hourly["datetime"]).dt.hour
average_power = hourly.groupby("hour")["ac_power"].mean()
plt.plot(average_power.index, average_power.values, marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power")
plt.title("Average AC Power by Hour")
plt.xticks(range(24))
plt.grid(True)
plt.savefig("results/figures/average_ac_power_by_hour.png")
plt.show()