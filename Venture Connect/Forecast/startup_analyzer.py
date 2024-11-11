import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StartupAnalyzer:
    def __init__(self):
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
        n_periods = len(dates)
        
        self.sample_data = {
            'Date': dates,
            'Revenue': [1000000 * (1.1 ** (i/12)) * (1 + 0.1 * np.sin(i/6)) for i in range(n_periods)],
            'Profit': [-500000 * (0.9 ** (i/12)) for i in range(n_periods)],
            'Expenditure': [1500000 * (1.08 ** (i/12)) for i in range(n_periods)],
            'Growth_Rate': [None] + [80 + 20 * np.sin(i/6) for i in range(n_periods-1)],
            'Valuation': [5000000 * (1.15 ** (i/12)) for i in range(n_periods)],
            'Cash_Burn': [120000 * (1.05 ** (i/12)) for i in range(n_periods)],
            'Market_Share': [0.5 * (1.1 ** (i/12)) for i in range(n_periods)],
            'Customer_Count': [1000 * (1.12 ** (i/12)) for i in range(n_periods)],
            'Customer_Acquisition_Cost': [500 * (0.95 ** (i/12)) for i in range(n_periods)],
            'Customer_Lifetime_Value': [2000 * (1.05 ** (i/12)) for i in range(n_periods)],
            'Churn_Rate': [5 * (0.98 ** (i/12)) for i in range(n_periods)],
            'Employee_Count': [50 * (1.08 ** (i/12)) for i in range(n_periods)],
            'Revenue_Per_Employee': None
        }
        
        self.df = pd.DataFrame(self.sample_data)
        self.df['Revenue_Per_Employee'] = self.df['Revenue'] / self.df['Employee_Count']
        
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

    def forecast_prophet(self, metric_name, periods=24):
        df_prophet = pd.DataFrame({
            'ds': self.df['Date'],
            'y': self.df[metric_name]
        })
        
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=False,
                       daily_seasonality=False,
                       seasonality_mode='multiplicative')
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def forecast_holtwinters(self, metric_name, periods=24):
        model = ExponentialSmoothing(
            self.df[metric_name],
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        ).fit()
        
        forecast = model.forecast(periods)
        return forecast

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

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Revenue & Profit Trends', 'Growth Metrics',
                          'Unit Economics', 'Operational Metrics',
                          'Forecasts', 'Key Performance Indicators'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                  [{"secondary_y": True}, {"secondary_y": True}],
                  [{"secondary_y": True}, {"type": "domain"}]]
        )

        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Revenue'],
                      name='Revenue', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Profit'],
                      name='Profit', line=dict(color='green')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Growth_Rate'],
                      name='Growth Rate', line=dict(color='red')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Customer_Lifetime_Value'],
                      name='LTV', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Customer_Acquisition_Cost'],
                      name='CAC', line=dict(color='orange')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=self.df['Date'], y=self.df['Revenue_Per_Employee'],
                      name='Revenue per Employee', line=dict(color='brown')),
            row=2, col=2
        )

        forecast = self.forecast_prophet('Revenue')
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                      name='Revenue Forecast', line=dict(dash='dash')),
            row=3, col=1
        )

        score = self.calculate_score()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score,
                gauge={'axis': {'range': [0, 10]},
                       'steps': [
                           {'range': [0, 3], 'color': "red"},
                           {'range': [3, 7], 'color': "yellow"},
                           {'range': [7, 10], 'color': "green"}
                       ]},
                title={'text': "Overall Score"}),
            row=3, col=2
        )

        fig.update_layout(height=1200, width=1000, title_text="Startup Analytics Dashboard")
        return fig

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

def analyze_startup():
    analyzer = StartupAnalyzer()
    
    metrics = analyzer.calculate_metrics()
    print("\nCurrent Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nForecasts for next 24 months:")
    metrics_to_forecast = ['Revenue', 'Profit', 'Valuation']
    for metric in metrics_to_forecast:
        forecast = analyzer.forecast_prophet(metric)
        print(f"\n{metric} Forecast (End of Period):")
        print(f"Expected: ${forecast['yhat'].iloc[-1]:,.2f}")
        print(f"Range: ${forecast['yhat_lower'].iloc[-1]:,.2f} to ${forecast['yhat_upper'].iloc[-1]:,.2f}")
    
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