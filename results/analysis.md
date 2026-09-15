Task 2 – Exploratory Plots

2.1 AC Power vs Irradiation

The plot shows a strong positive relationship: as irradiation increases, AC power also increases almost linearly. This agrees with the physics because higher sunlight provides more energy to the solar panels, producing more power. A few points deviate from the main trend, especially at higher irradiation levels, but overall the data agrees well with the expected behavior.

2.2 Module Temperature vs Ambient Temperature

Module temperature generally increases as ambient temperature increases, showing a positive relationship between the two. The data agrees with the expected physics, although the points are more spread out at higher temperatures, likely because irradiation also affects module temperature.

2.3 AC Power vs DC Power

AC power and DC power show an almost perfectly linear relationship, with AC power increasing as DC power increases. This agrees with the expected physics because the inverter converts DC power from the panels into AC power with relatively consistent efficiency. The very small deviation from the straight-line trend suggests that the conversion efficiency remains fairly stable.

Comment on the AC/DC Ratio

The AC/DC power ratio stays nearly constant across the data, showing that the inverter operates with fairly consistent efficiency. Small variations in the ratio may be due to inverter losses and measurement differences.

2.4 Average AC Power per Hour

Average AC power is close to zero during the night and increases after sunrise, reaching its maximum around 11–12 PM before decreasing toward sunset. This matches the expected solar pattern because solar power depends on sunlight availability.

Task 5 – Analysis

5.1 Normal Equation Weights for Set A

Irradiation has the largest weight (θ₁ ≈ 8258.30), showing that it has the strongest effect on predicted AC power. Its positive sign agrees with the physics because higher irradiation produces more solar power. The negative temperature weights are reasonable because higher panel temperature can reduce solar panel efficiency, while the hour terms capture the daily solar pattern.

5.2 Set B Compared with Set A

Set B has a daytime RMSE of 3417.65 kW, while Set A has a daytime RMSE of 723.41 kW. Therefore, Set B is 2694.24 kW worse, which is about 9.86% of the plant's peak hourly power. Since Set B's daytime RMSE is about 12.51% of the peak power, public weather data is less accurate than on-site sensors, so it may be useful for a rooftop installer for rough estimates, but it is not as reliable for accurate power prediction.

Set B's daytime RMSE is about 372.4% higher than Set A's.

5.3 Comparison of the Three Solvers

The Batch Gradient Descent result is almost the same as the Normal Equation result, with a maximum θ difference of only 0.00014. The SGD result is different, especially for the temperature weights, with a maximum difference of about 845.24, so the three solvers did not reach exactly the same θ.

Batch GD used 50,000 iterations with α = 0.0001, while SGD used 50 epochs with α = 0.01. For this dataset, I would choose Normal Equation or Batch GD because the dataset is relatively small and Batch GD gives almost the same solution; for a dataset with 10 million rows, Gradient Descent/SGD would be more suitable because the Normal Equation requires expensive matrix calculations.

5.4 Batch vs Stochastic Gradient Descent

The Batch Gradient Descent curve is smoother because it calculates the gradient using the whole training dataset before each update. SGD updates the weights one sample at a time, so its cost can fluctuate more, although in this plot the SGD curves appear relatively smooth because the cost is recorded after each complete epoch.

5.5 Residuals vs Hour of Day

The model has the largest residuals around hours 10–15, especially around 11 AM, where the residual is strongly positive. This may be caused by changes in irradiation, cloud cover, or panel temperature that the model does not fully capture during peak solar hours.