import pandas as pd
import numpy as np

class StartupDataGenerator:
    
    def __init__(self, start_date='2019-01-01', end_date='2023-12-31', freq='QE'):

        self.dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        self.n_periods = len(self.dates)
        
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
    def generate_base_metrics(self):
        return {
            'Date': self.dates,
            'Revenue': self._generate_revenue(),
            'Profit': self._generate_profit(),
            'Expenditure': self._generate_expenditure(),
            'Growth_Rate': self._generate_growth_rate(),
            'Valuation': self._generate_valuation(),
            'Cash_Burn': self._generate_cash_burn(),
            'Market_Share': self._generate_market_share(),
            'Customer_Count': self._generate_customer_metrics()['customers'],
            'Customer_Acquisition_Cost': self._generate_customer_metrics()['cac'],
            'Customer_Lifetime_Value': self._generate_customer_metrics()['ltv'],
            'Churn_Rate': self._generate_customer_metrics()['churn'],
            'Employee_Count': self._generate_employee_count()
        }
    
    def _add_noise(self, series, noise_level=0.05):
        """Add realistic noise to the series"""
        noise = np.random.normal(0, noise_level, len(series))
        return series * (1 + noise)
    
    def _generate_revenue(self):
        """Generate revenue with seasonal patterns and realistic variations"""
        base_revenue = [1000000 * (1.1 ** (i/4)) * (1 + 0.1 * np.sin(i/2)) 
                       for i in range(self.n_periods)]
        return self._add_noise(base_revenue)
    
    def _generate_profit(self):
        """Generate profit data with gradual improvement"""
        base_profit = [-500000 * (0.9 ** (i/4)) for i in range(self.n_periods)]
        # Add tendency toward profitability in later periods
        profit_improvement = [min(0, p * (1 + 0.02 * i)) for i, p in enumerate(base_profit)]
        return self._add_noise(profit_improvement, 0.08)
    
    def _generate_expenditure(self):
        """Generate expenditure data with quarterly variation"""
        base_expenditure = [1500000 * (1.08 ** (i/4)) for i in range(self.n_periods)]
        # Add quarterly seasonality to expenditure
        seasonal_expenditure = [e * (1 + 0.05 * np.sin(i/2)) for i, e in enumerate(base_expenditure)]
        return self._add_noise(seasonal_expenditure)
    
    def _generate_growth_rate(self):
        """Generate growth rate with seasonal variation and trend"""
        base_growth = [None] + [80 + 20 * np.sin(i/2) - 0.5 * i 
                               for i in range(self.n_periods-1)]
        return base_growth  # No noise added to growth rate
    
    def _generate_valuation(self):
        """Generate company valuation with market sentiment effects"""
        base_valuation = [5000000 * (1.15 ** (i/4)) for i in range(self.n_periods)]
        market_sentiment = [1 + 0.15 * np.sin(i/3) for i in range(self.n_periods)]
        return self._add_noise([v * s for v, s in zip(base_valuation, market_sentiment)])
    
    def _generate_cash_burn(self):
        """Generate cash burn rate with efficiency improvements"""
        base_burn = [120000 * (1.05 ** (i/4)) for i in range(self.n_periods)]
        efficiency_factor = [1 / (1 + 0.02 * i) for i in range(self.n_periods)]
        return self._add_noise([b * e for b, e in zip(base_burn, efficiency_factor)])
    
    def _generate_market_share(self):
        """Generate market share data with competitive effects"""
        base_share = [0.5 * (1.1 ** (i/4)) for i in range(self.n_periods)]
        competitive_pressure = [1 - 0.01 * i for i in range(self.n_periods)]
        return self._add_noise([s * c for s, c in zip(base_share, competitive_pressure)])
    
    def _generate_customer_metrics(self):
        """Generate customer-related metrics with interrelated patterns"""
        customers = [1000 * (1.12 ** (i/4)) for i in range(self.n_periods)]
        customers = self._add_noise(customers)
        
        # CAC increases as customer base grows
        cac = [500 * (0.95 ** (i/4)) * (1 + 0.001 * (i/4)) for i in range(self.n_periods)]
        cac = self._add_noise(cac)
        
        # LTV improves with scale
        ltv = [2000 * (1.05 ** (i/4)) * (1 + 0.002 * (i/4)) for i in range(self.n_periods)]
        ltv = self._add_noise(ltv)
        
        # Churn improves with maturity
        churn = [5 * (0.98 ** (i/4)) for i in range(self.n_periods)]
        churn = self._add_noise(churn, 0.03)
        
        return {
            'customers': customers,
            'cac': cac,
            'ltv': ltv,
            'churn': churn
        }
    
    def _generate_employee_count(self):
        base_count = [50 * (1.08 ** (i/4)) for i in range(self.n_periods)]
        hiring_waves = [1 + 0.1 * (i % 4 == 0) for i in range(self.n_periods)]
        return self._add_noise([c * h for c, h in zip(base_count, hiring_waves)])
    
    def generate_dataset(self):
        data = self.generate_base_metrics()
        if not data: 
            raise ValueError("The generated base metrics are empty.")
        df = pd.DataFrame(data)
        df['Revenue_Per_Employee'] = df['Revenue'] / df['Employee_Count']
        return df 

    def save_to_csv(self, filename="startup_data.csv"):
        df = self.generate_dataset()
        if df is None or df.empty:  
            raise ValueError("The generated DataFrame is empty. Cannot save to CSV.")
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

