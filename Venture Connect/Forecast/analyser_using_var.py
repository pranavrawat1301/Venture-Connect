from data_generation_for_analyser import StartupDataGenerator
from visualizer import StartupVisualizer

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StartupAnalyzer:
    
    def __init__(self, data=None):

        if data is None:
            data_generator = StartupDataGenerator()
            self.df = data_generator.generate_dataset()
            print("Generating Custom Data")
            data_generator.save_to_csv()
        else:
            self.df = data.copy()
    
    def forecast_var(self, metrics_to_forecast=['Revenue', 'Profit', 'Valuation'], periods=8):

        data_for_var = self.df[metrics_to_forecast].copy()
        

        min_values = data_for_var.min()
        constants = {col: abs(min(0, val)) + 1 for col, val in min_values.items()}
        log_data = np.log(data_for_var + pd.Series(constants))
        
        n_obs = len(log_data)
        n_vars = len(metrics_to_forecast)
        
        # VAR model needs n_vars^2 * lags + n_vars parameters
        # need at least 2 observations per parameter for estimation
        max_possible_lags = (n_obs - n_vars) // (2 * n_vars * n_vars)
        max_lags = max(1, min(4, max_possible_lags))  
        
        try:
            model = VAR(log_data)
            
            if max_lags == 1:
                selected_order = 1
            else:
                order = model.select_order(maxlags=max_lags)
                selected_order = min(order.selected_orders['aic'], max_lags)
            
            model_fitted = model.fit(selected_order)
            
            # Generate forecasts in log scale
            log_forecast = model_fitted.forecast(log_data.values, steps=periods)
            
            constants_array = np.array([constants[metric] for metric in metrics_to_forecast])
            forecast = np.exp(log_forecast) - constants_array.reshape(1, -1)
            
            future_dates = pd.date_range(
                start=self.df['Date'].iloc[-1] + pd.Timedelta(days=90),
                periods=periods,
                freq='Q'
            )
            
            forecast_df = pd.DataFrame(
                forecast,
                columns=metrics_to_forecast,
                index=future_dates
            )
            
            # Calculate confidence intervals
            confidence_intervals = {}
            fitted_values = pd.DataFrame(model_fitted.fittedvalues, columns=metrics_to_forecast)
            
            for i, metric in enumerate(metrics_to_forecast):
                # Calculate residuals in original scale
                actual = data_for_var[metric]
                fitted = np.exp(fitted_values[metric]) - constants[metric]
                residuals = actual - fitted
                
                # Calculate prediction error that increases with forecast horizon
                base_std_err = np.std(residuals)
                forecast_std_err = np.array([
                    base_std_err * np.sqrt(1 + h * 0.15) for h in range(1, periods + 1)
                ])
                
                # Add confidence intervals
                forecast_values = forecast_df[metric].values
                confidence_intervals[f'{metric}_lower'] = forecast_values - 1.96 * forecast_std_err
                confidence_intervals[f'{metric}_upper'] = forecast_values + 1.96 * forecast_std_err
                
                # Ensure lower bounds make sense for the metric
                if metric in ['Revenue', 'Valuation', 'Customer_Count']:
                    confidence_intervals[f'{metric}_lower'] = np.maximum(0, confidence_intervals[f'{metric}_lower'])
            
            for col, values in confidence_intervals.items():
                forecast_df[col] = values
            
            for metric in metrics_to_forecast:
                actual = data_for_var[metric]
                fitted = np.exp(fitted_values[metric]) - constants[metric]
                residuals = actual - fitted
                rmse = np.sqrt(np.mean(residuals**2))
                forecast_df[f'{metric}_rmse'] = rmse
            
            forecast_df.reset_index(inplace=True)
            forecast_df.rename(columns={'index': 'Date'}, inplace=True)
            
            print(f"\nVAR Model Information:")
            print(f"Number of observations: {n_obs}")
            print(f"Number of variables: {n_vars}")
            print(f"Selected lag order: {selected_order}")
            print(f"Maximum possible lags: {max_possible_lags}")
            
            return forecast_df
        
        except Exception as e:
            print(f"Error in VAR forecasting: {str(e)}")
            print("\nDiagnostic information:")
            print(f"Number of observations: {n_obs}")
            print(f"Number of variables: {n_vars}")
            print(f"Maximum possible lags: {max_possible_lags}")
            raise

    def calculate_metrics(self):
        metrics = {}
        latest = self.df.iloc[-1]
        
        # Profitability metrics
        metrics['profit_margin'] = (latest['Profit'] / latest['Revenue']) * 100
        metrics['burn_rate'] = latest['Cash_Burn']
        metrics['runway_months'] = abs(latest['Cash_Burn'] * 12 / latest['Profit']) if latest['Profit'] < 0 else float('inf')
        
        # Growth metrics
        metrics['revenue_growth'] = ((self.df['Revenue'].iloc[-1] / self.df['Revenue'].iloc[0]) - 1) * 100
        metrics['customer_growth'] = ((self.df['Customer_Count'].iloc[-1] / self.df['Customer_Count'].iloc[0]) - 1) * 100
        
        # Efficiency metrics
        metrics['revenue_per_customer'] = latest['Revenue'] / latest['Customer_Count']
        metrics['revenue_per_employee'] = latest['Revenue_Per_Employee']
        
        # Unit economics
        metrics['ltv_cac_ratio'] = latest['Customer_Lifetime_Value'] / latest['Customer_Acquisition_Cost']
        metrics['churn_rate'] = latest['Churn_Rate']
        
        return metrics

    def calculate_score(self):
        metrics = self.calculate_metrics()
        
        scores = {
            'profit_margin': {
                'weight': 0.15,
                'score': min(10, max(1, (metrics['profit_margin'] + 20) / 5))
            },
            'revenue_growth': {
                'weight': 0.15,
                'score': min(10, max(1, metrics['revenue_growth'] / 20))
            },
            'customer_growth': {
                'weight': 0.1,
                'score': min(10, max(1, metrics['customer_growth'] / 20))
            },
            'burn_rate': {
                'weight': 0.1,
                'score': min(10, max(1, 10 - (metrics['burn_rate'] / 100000)))
            },
            'ltv_cac_ratio': {
                'weight': 0.2,
                'score': min(10, max(1, metrics['ltv_cac_ratio']))
            },
            'revenue_per_employee': {
                'weight': 0.15,
                'score': min(10, max(1, metrics['revenue_per_employee'] / 100000))
            },
            'churn_rate': {
                'weight': 0.15,
                'score': min(10, max(1, 10 - metrics['churn_rate']))
            }
        }
        
        final_score = sum(s['weight'] * s['score'] for s in scores.values())
        return round(final_score, 1)

    def create_visualizations(self):
        visualizer = StartupVisualizer(self)
        return visualizer.create_visualizations()

    def get_improvement_areas(self):
        metrics = self.calculate_metrics()
        improvements = []
        
        if metrics['profit_margin'] < 0:
            improvements.append({
                'area': 'Profitability',
                'severity': 'High',
                'suggestion': 'Focus on reducing operational costs and improving pricing strategy',
                'specific_actions': [
                    'Conduct cost structure analysis',
                    'Implement value-based pricing',
                    'Optimize resource allocation'
                ]
            })
            
        if metrics['ltv_cac_ratio'] < 3:
            improvements.append({
                'area': 'Unit Economics',
                'severity': 'Medium',
                'suggestion': 'Improve customer lifetime value to acquisition cost ratio',
                'specific_actions': [
                    'Enhance customer retention programs',
                    'Optimize marketing spend',
                    'Develop upselling strategies'
                ]
            })
            
        if metrics['revenue_per_employee'] < 200000:
            improvements.append({
                'area': 'Operational Efficiency',
                'severity': 'Medium',
                'suggestion': 'Increase revenue per employee',
                'specific_actions': [
                    'Automate manual processes',
                    'Review team structure and roles',
                    'Implement productivity tools'
                ]
            })
            
        return improvements

def analyze_startup(data=None):
    analyzer = StartupAnalyzer(data)
    
    metrics = analyzer.calculate_metrics()
    print("\nCurrent Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nForecasts for next 8 quarters:")
    forecast = analyzer.forecast_var()
    for metric in ['Revenue', 'Profit', 'Valuation']:
        print(f"\n{metric} Forecast (End of Period):")
        print(f"Expected: ${forecast[metric].iloc[-1]:,.2f}")
        print(f"Range: ${forecast[f'{metric}_lower'].iloc[-1]:,.2f} to ${forecast[f'{metric}_upper'].iloc[-1]:,.2f}")
    
    score = analyzer.calculate_score()
    print(f"\nOverall Startup Score (1-10): {score}")
    
    improvements = analyzer.get_improvement_areas()
    print("\nAreas for Improvement:")
    for imp in improvements:
        print(f"\n{imp['area']} (Severity: {imp['severity']}):")
        print(f"Suggestion: {imp['suggestion']}")
        print("Specific Actions:")
        for action in imp['specific_actions']:
            print(f"- {action}")
    
    fig = analyzer.create_visualizations()
    fig.show()

if __name__ == "__main__":
    analyze_startup()