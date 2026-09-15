import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression import hypothesis, cost, fit_normal, fit_batch_gd, fit_sgd, rmse

# Load and combine plant and weather data
data = pd.read_csv("data/plant1_hourly.csv")
weather = pd.read_csv("data/plant1_openmeteo.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])
data = pd.merge(data, weather, on="datetime", how="inner").dropna()
data = data.sort_values("datetime").reset_index(drop=True)

# Time features
data["hour"] = data["datetime"].dt.hour
data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)

# Split by date
train = data[data["datetime"] < "2020-06-11"]
test = data[data["datetime"] >= "2020-06-11"]
print("Train rows:", len(train), "| Test rows:", len(test))
features = {
    "A": ["irradiation", "module_temp", "ambient_temp", "sin_hour", "cos_hour"],
    "B": ["sw_radiation", "temp_2m", "cloud_cover", "sin_hour", "cos_hour"],
}
target = "ac_power"
y_train = train[target].values
y_test = test[target].values

#Scale with training statistics, add intercept
def add_intercept(X):
    return np.column_stack((np.ones(len(X)), X))
X_train, X_test, scaling = {}, {}, {}
for name, cols in features.items():
    mean = train[cols].values.mean(axis=0)
    std = train[cols].values.std(axis=0)
    X_train[name] = add_intercept((train[cols].values - mean) / std)
    X_test[name] = add_intercept((test[cols].values - mean) / std)

    scaling[name] = {"features": cols, "mean": mean.tolist(), "std": std.tolist()}

#Choose the learning rate (Set A, training rows only)
batch_alphas = [1e-5, 1e-4, 1e-3]
sgd_alphas = [1e-4, 1e-3, 1e-2]
plt.figure(figsize=(8, 5))
print("\nBatch GD learning-rate search (Set A, 500 iterations)")
for alpha in batch_alphas:
    _, J = fit_batch_gd(X_train["A"], y_train, alpha, 500)
    decreasing = bool(np.all(np.diff(J) <= 0))
    print(f"  alpha={alpha:g}  final J={J[-1]:.4e}  decreases every iteration: {decreasing}")
    plt.plot(range(1, 501), J, label=f"alpha = {alpha:g}")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("J(theta)  (log scale)")
plt.title("Batch gradient descent, Set A")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/batch_gd_learning_rates.png")
plt.close()
plt.figure(figsize=(8, 5))
print("\nSGD learning-rate search (Set A, 50 epochs)")
for alpha in sgd_alphas:
    _, J = fit_sgd(X_train["A"], y_train, alpha, 50)
    print(f"  alpha={alpha:g}  final J={J[-1]:.4e}")
    plt.plot(range(1, 51), J, label=f"alpha = {alpha:g}")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("J(theta)  (log scale)")
plt.title("Stochastic gradient descent, Set A")
plt.legend()
plt.tight_layout()
plt.savefig("results/figures/sgd_learning_rates.png", dpi=150)
plt.close()

# Read from the plots above:
#   batch: 1e-5 too small, 1e-3 too large (J blows up), 1e-4 about right
#   SGD:   1e-4 too small, 1e-3 still slow, 1e-2 about right
BATCH_ALPHA = 1e-4
SGD_ALPHA = 1e-2

# Enough iterations for batch GD to reach the normal-equation theta
BATCH_ITERS = 50000
SGD_EPOCHS = 50

#Fit all three methods on Set A and Set B
thetas = {}
for name in features:
    X, y = X_train[name], y_train
    theta_normal = fit_normal(X, y)
    theta_batch, J_batch = fit_batch_gd(X, y, BATCH_ALPHA, BATCH_ITERS)
    theta_sgd, J_sgd = fit_sgd(X, y, SGD_ALPHA, SGD_EPOCHS)
    thetas[name] = {"normal": theta_normal, "batch": theta_batch, "sgd": theta_sgd}
    print(f"\nSet {name}")
    print("  J(theta) normal eq:", cost(X, y, theta_normal))

    # Once converged, J only changes by floating-point rounding (~1e-15 relative)
    tolerance = 1e-12 * J_batch[-1]
    print("  J decreases every batch GD iteration:",
          bool(np.all(np.diff(J_batch) <= tolerance)))
    print("  Max |theta_batch - theta_normal|:", np.max(np.abs(theta_batch - theta_normal)))
    print("  Max |theta_sgd   - theta_normal|:", np.max(np.abs(theta_sgd - theta_normal)))

#Predict on test rows, clip at zero, RMSE
daytime = test["irradiation"].values > 0
solver_labels = {
    "normal": "Normal equation",
    "batch": f"Batch GD (alpha={BATCH_ALPHA:g}, {BATCH_ITERS} iters)",
    "sgd": f"Stochastic GD (alpha={SGD_ALPHA:g}, {SGD_EPOCHS} epochs)",
}
rows = []
for name in features:
    for solver, label in solver_labels.items():
        pred = np.maximum(hypothesis(X_test[name], thetas[name][solver]), 0)
        rows.append({
            "Solver": label,
            "Features": f"Set {name}",
            "All hours": round(rmse(y_test, pred), 2),
            "Daytime only": round(rmse(y_test[daytime], pred[daytime]), 2),
        })
table2 = pd.DataFrame(rows)
table2.to_csv("results/table/table2_test_rmse.csv", index=False)
print("\nTable 2 - Test-set RMSE (kW)")
print(table2.to_string(index=False))
peak = train[target].max()
print(f"\nPeak hourly ac_power in training data: {peak:.2f} kW")

# Table 3 - learned theta for Set A
theta_names = ["theta0 intercept", "theta1 irradiation", "theta2 module_temp",
               "theta3 ambient_temp", "theta4 sin hour", "theta5 cos hour"]
table3 = pd.DataFrame({
    "Normal eq.": thetas["A"]["normal"],
    "Batch GD": thetas["A"]["batch"],
    "SGD": thetas["A"]["sgd"],
}, index=theta_names)
table3.loc["Max |theta_GD - theta_normal|"] = [
    np.nan,
    np.max(np.abs(thetas["A"]["batch"] - thetas["A"]["normal"])),
    np.max(np.abs(thetas["A"]["sgd"] - thetas["A"]["normal"])),
]
table3.to_csv("results/table/table3_theta_set_A.csv")
print("\nTable 3 - Learned theta (Set A)")
print(table3.to_string(float_format=lambda v: f"{v:.4f}"))

# Save weights and scaling statistics for the front end 
weights = {
    name: {**scaling[name], "theta": thetas[name]["normal"].tolist()}
    for name in features
}
with open("results/weights.json", "w") as f:
    json.dump(weights, f, indent=2)

print("\nSaved figures to results/figures/, tables to results/table/, weights.json to results/")

# Figure 5 - actual vs predicted AC power on the test week (normal equation)
pred_A = np.maximum(hypothesis(X_test["A"], thetas["A"]["normal"]), 0)
pred_B = np.maximum(hypothesis(X_test["B"], thetas["B"]["normal"]), 0)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
axes[0].plot(test["datetime"], y_test, label="Actual", color="black", linewidth=1.5)
axes[0].plot(test["datetime"], pred_A, label="Predicted (Set A)", color="tab:blue", linestyle="--")
axes[0].set_title("Set A: on-site sensors")
axes[0].set_ylabel("AC power (kW)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(test["datetime"], y_test, label="Actual", color="black", linewidth=1.5)
axes[1].plot(test["datetime"], pred_B, label="Predicted (Set B)", color="tab:orange", linestyle="--")
axes[1].set_title("Set B: public weather only")
axes[1].set_ylabel("AC power (kW)")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True)

fig.suptitle("Actual vs predicted AC power, test week (11-17 June), normal equation")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("results/figures/actual_vs_predicted_test_week.png", dpi=150)
plt.close()

#residuals vs hour of day, Set A normal equation, test week
pred_A = np.maximum(hypothesis(X_test["A"], thetas["A"]["normal"]), 0)
residuals = y_test - pred_A
hours = test["hour"].values

# Average residual for each hour of the day
mean_residual = pd.Series(residuals).groupby(hours).mean()
plt.figure(figsize=(10, 5))
plt.scatter(hours, residuals, alpha=0.4, label="Each test hour")
plt.plot(mean_residual.index, mean_residual.values, color="red", marker="o", label="Mean residual")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Hour of day")
plt.ylabel("Residual  y - h(x)  (kW)")
plt.title("Residuals vs hour of day, Set A normal equation (test week)")
plt.xticks(range(24))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/residuals_vs_hour.png", dpi=150)
plt.close()
