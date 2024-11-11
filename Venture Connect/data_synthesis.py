import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DataSynthesizer:
    def __init__(self):
        self.industries = [
            "Technology & Software","Sustainability & Environment","Healthcare & Wellness","Agriculture & Food Tech",
            "Finance & FinTech","Education & EdTech","Real Estate & Property Tech","Transportation & Mobility",
            "Cybersecurity","E-commerce & Retail","Data & Analytics","Artificial Intelligence & Machine Learning",
            "Energy & Clean Tech","Construction & Infrastructure","Creative & Design"
        ]
        
        self.locations = ['US', 'EU', 'Asia', 'UK', 'LatAm']
        self.funding_stages = ['Seed', 'Series A', 'Series B', 'Series C']
        
    def generate_startup_data(self, num_startups: int = 40) -> pd.DataFrame:
    
        return pd.DataFrame({
            'id': range(1, num_startups + 1),
            'name': [f'Startup_{i}' for i in range(1, num_startups + 1)],
            'industry': np.random.choice(self.industries, num_startups),
            'valuation': np.random.uniform(1e5, 1e9, num_startups),
            'profitability': np.random.uniform(-0.3, 0.5, num_startups),
            'avg_price': np.random.uniform(10, 5000, num_startups),
            'team_size': np.random.randint(2, 500, num_startups),
            'equity_offered': np.random.uniform(0.05, 0.3, num_startups),
            'growth_rate': np.random.uniform(0.1, 2.0, num_startups),
            'founding_year': np.random.randint(2015, 2024, num_startups),
            'monthly_revenue': np.random.uniform(1e4, 1e7, num_startups),
            'burn_rate': np.random.uniform(1e4, 5e5, num_startups),
            'funding_stage': np.random.choice(self.funding_stages, num_startups),
            'location': np.random.choice(self.locations, num_startups),
            'b2b': np.random.choice([True, False], num_startups)
        })
    
    def generate_investor_data(self, num_investors: int = 50) -> pd.DataFrame:
        # Helper function to get unique random samples
        def get_unique_preferences(options: List[str], min_count: int, max_count: int) -> List[str]:
            count = np.random.randint(min_count, min(max_count + 1, len(options) + 1))
            return list(np.random.choice(options, size=count, replace=False))
        
        return pd.DataFrame({
            'id': range(1, num_investors + 1),
            'name': [f'Investor_{i}' for i in range(1, num_investors + 1)],
            'preferred_industry': [get_unique_preferences(self.industries, 1, 3) 
                                 for _ in range(num_investors)],
            'min_valuation': np.random.uniform(1e5, 5e8, num_investors),
            'max_valuation': np.random.uniform(5e8, 2e9, num_investors),
            'target_profitability': np.random.uniform(-0.2, 0.4, num_investors),
            'preferred_team_size': np.random.randint(2, 500, num_investors),
            'target_equity': np.random.uniform(0.05, 0.3, num_investors),
            'min_growth_rate': np.random.uniform(0.1, 1.0, num_investors),
            'preferred_stages': [get_unique_preferences(self.funding_stages, 1, 3)
                               for _ in range(num_investors)],
            'preferred_locations': [get_unique_preferences(self.locations, 1, 3)
                                  for _ in range(num_investors)],
            'investment_history': [np.random.randint(5, 50) for _ in range(num_investors)],
            'avg_check_size': np.random.uniform(1e5, 1e7, num_investors),
            'b2b_focus': np.random.choice([True, False, None], num_investors)
        })
    
    def save_data(self, startups_df: pd.DataFrame, investors_df: pd.DataFrame, 
                  startups_file: str = 'startups.csv', 
                  investors_file: str = 'investors.csv'):
        
        startups_df.to_csv(startups_file, index=False)
        investors_df.to_csv(investors_file, index=False)
        
    def load_data(self, startups_file: str = 'startups.csv', 
                  investors_file: str = 'investors.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:

        startups_df = pd.read_csv(startups_file)
        investors_df = pd.read_csv(investors_file)
        return startups_df, investors_df