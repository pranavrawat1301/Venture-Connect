# Venture Connect

## Overview

This repository contains a Python-based system for both startup & investor recommendation and startup performance analysis. It integrates machine learning techniques and data synthesis to generate synthetic datasets for startups and investors, followed by building a recommender system for personalized recommendations. Additionally, it analyzes startup data to forecast financial metrics and evaluate overall performance.

## Features

### Startup & Investor Recommendation System

- **Data Synthesis**: Generates synthetic data for startups and investors.
- **Recommender System**: Uses machine learning models to provide top startup-investor matches based on various criteria.
- **Preprocessing**: Scales numerical features and calculates similarity matrices for industry and other metrics.
- **Customization**: Supports custom feature weights for personalized recommendations.
- **Files & Modules**:
  - `data.py`: Main file for generating and saving data.
  - `data_synthesis.py`: Contains the `DataSynthesizer` class for generating synthetic startup and investor data.
  - `recommendor.py`: Implements the recommendation logic, including similarity calculations and matching.
  - `recommendation.py`: Provides a CLI interface for users to input IDs and receive top matches.
- **Setup**: Instantiates `StartupInvestorRecommender` with the generated data and feature weights for personalized recommendations.

### Startup Performance Analyzer

- **Data Generation**: Utilizes a `StartupDataGenerator` to create synthetic startup data with realistic noise, seasonal patterns, and trend-based growth.
- **Data Analysis**: Employs VAR (Vector AutoRegression) models to forecast key metrics like Revenue, Profit, and Valuation over future periods.
- **Visualization**: Provides visual representations of startup performance, such as trend lines, confidence intervals, and comparative metrics.
- **Metrics Calculation**: Computes various metrics such as Profit Margin, Growth Rates, Unit Economics, and Efficiency, providing a comprehensive assessment of startup health.
- **Improvement Insights**: Suggests areas for improvement based on performance metrics and forecasts, such as profitability, revenue growth, and operational efficiency.
- **Machine Learning Integration**: Implements predictive models to enhance forecasting accuracy and model robustness, including feature importance analysis through methods like `ExtraTreesClassifier`.
- **Dependencies**: Utilizes libraries such as pandas, numpy, sklearn, statsmodels, and plotly for data manipulation, statistical modeling, and visualization.

## Usage

### Startup & Investor Recommendation System

1. **Data Synthesis**: Use `DataSynthesizer` to generate datasets for startups and investors.
2. **Recommendation**: Utilize `StartupInvestorRecommender` with feature weights to generate personalized matches.

### Startup Performance Analyzer

1. **Data Analysis**: Analyze startup data using `analyze_startup(data=custom_data)` to get performance insights.
   - Generates forecasts for the next 8 quarters, evaluates current metrics, and provides improvement suggestions.
2. **Data Generation**: The `StartupDataGenerator` class generates synthetic data to simulate real-world startup performance.
3. **Custom Data Handling**: Optionally, custom startup datasets can be provided and analyzed through the `analyze_startup` function.

## Visualization

- The analysis includes visualizations created using Plotly, which illustrate metrics, forecasts, and confidence intervals.

---

## Snapshots of the result

# Recommendation
![image](https://github.com/user-attachments/assets/a18f9e3a-fb99-453d-b750-5c7890eb6ab6)

# Analysis and Forecast Plot
![image](https://github.com/user-attachments/assets/4d67770c-b159-4c6f-8ff7-33c41c0947ee)
