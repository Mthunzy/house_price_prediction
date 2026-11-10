# House Price Prediction | Regression - Capstone Project

## Author
**Mthunzi Ntombela**  
Johannesburg, Gauteng | 083 981 6871 | m2nzhiey@gmail.com  
GitHub: https://github.com/Mthunzy  
LinkedIn: linkedin.com/in/mthunzi-ntombela-51646b145

---

## The Problem
How much is a house worth?

Estate agents guess. Buyers overpay. Sellers underprice.

Can we predict house price accurately from data like location, rooms, and income?

---

## The Dataset
- **Source:** California Housing Dataset
- **Size:** 20,640 houses
- **Target:** `median_house_value` - The price we predict
- **Features:** Longitude, Latitude, Housing Age, Total Rooms, Total Bedrooms, Population, Households, Median Income

---

## What I Did
1. Loaded 20,640 houses with Pandas
2. Explored data, checked missing values and price distribution
3. Cleaned data - removed missing values
4. Split data 80% Train / 20% Test
5. **Model 1:** Linear Regression (Baseline Model)
6. **Model 2:** Random Forest Regressor (Advanced Model)
7. Compared both models with MAE, RMSE and R2 Score
8. Analysed Feature Importance to find what drives price
9. Visualised Feature Importance with Matplotlib

---

## Results

| Model | R2 Score | What It Means |
| :--- | :--- | :--- |
| Linear Regression | See output | Baseline - simple linear relationship |
| **Random Forest** | **See output - Higher** | **Winner - Captures complex patterns** |

Random Forest beats Linear Regression because house prices are NOT linear - location and income interact in complex ways.

---

## Key Findings
- **Location is #1** - Latitude / Longitude drives price most
- **Income is #2** - Richer area = Higher house price
- **House age matters** - Newer houses tend to cost more
- Estate agents can price houses in seconds, not days
- Buyers can check if a house is overpriced before buying

---

## Tools Used
Python | Pandas | Scikit-Learn | Linear Regression | Random Forest Regressor | Matplotlib | Regression

---

## Project Structure
