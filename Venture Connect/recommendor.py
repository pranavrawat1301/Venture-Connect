from data_synthesis import DataSynthesizer

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class StartupInvestorRecommender:
    def __init__(self, startups_df: pd.DataFrame, investors_df: pd.DataFrame, 
                 feature_weights: Dict[str, float] = None):

        self.startups_data = startups_df
        self.investors_data = investors_df
        
        # Default feature weights
        self.feature_weights = feature_weights or {
            'industry': 0.25,
            'valuation': 0.2,
            'profitability': 0.15,
            'team_size': 0.1,
            'equity': 0.15,
            'growth_rate': 0.15
        }
        
        self.startup_scaler = MinMaxScaler()
        self.investor_scaler = MinMaxScaler()
        
        self.industries = list(set(self.startups_data['industry'].unique()))
    
    def calculate_stage_compatibility(self, startup_stage: str, investor_stages: List[str]) -> float:

        if startup_stage in investor_stages:
            return 1.0
        
        stage_order = ['Seed', 'Series A', 'Series B', 'Series C']
        startup_idx = stage_order.index(startup_stage)
        investor_idxs = [stage_order.index(stage) for stage in investor_stages]
        
        min_distance = min(abs(startup_idx - idx) for idx in investor_idxs)
        return 1.0 / (1.0 + min_distance)
    
    def calculate_growth_score(self, startup_growth: float, investor_min_growth: float) -> float:

        if startup_growth >= investor_min_growth:
            return 1.0
        return startup_growth / investor_min_growth
    
    def calculate_location_score(self, startup_location: str, investor_locations: List[str]) -> float:
        
        return 1.0 if startup_location in investor_locations else 0.5
    
    def preprocess_data(self):
        
        startup_industry_encoded = pd.get_dummies(self.startups_data['industry'])
        
        investor_industry_matrix = np.zeros((len(self.investors_data), len(self.industries)))
        for idx, industries in enumerate(self.investors_data['preferred_industry']):
            if isinstance(industries, str): 
                industries = eval(industries)
            for industry in industries:
                if industry in self.industries:
                    industry_idx = self.industries.index(industry)
                    investor_industry_matrix[idx, industry_idx] = 1
                    
        investor_industry_encoded = pd.DataFrame(
            investor_industry_matrix, 
            columns=[f'industry_{ind}' for ind in self.industries]
        )
        
        startup_numerical = self.startups_data[['valuation', 'profitability', 
                                              'team_size', 'equity_offered', 'growth_rate']]
        startup_numerical_scaled = self.startup_scaler.fit_transform(startup_numerical)
        
        investor_numerical = self.investors_data[['max_valuation', 'target_profitability', 
                                                'preferred_team_size', 'target_equity', 'min_growth_rate']]
        investor_numerical_scaled = self.investor_scaler.fit_transform(investor_numerical)
        
        self.startup_features = np.hstack([
            startup_numerical_scaled * self.feature_weights['valuation'],
            startup_industry_encoded.values * self.feature_weights['industry']
        ])
        
        self.investor_features = np.hstack([
            investor_numerical_scaled * self.feature_weights['valuation'],
            investor_industry_matrix * self.feature_weights['industry']
        ])
    
    def calculate_similarity(self) -> np.ndarray:
        
        base_similarity = cosine_similarity(self.startup_features, self.investor_features)
        
        additional_factors = np.zeros_like(base_similarity)
        
        for i, startup in self.startups_data.iterrows():
            for j, investor in self.investors_data.iterrows():
                
                investor_stages = eval(investor['preferred_stages']) if isinstance(investor['preferred_stages'], str) else investor['preferred_stages']
                investor_locations = eval(investor['preferred_locations']) if isinstance(investor['preferred_locations'], str) else investor['preferred_locations']
                
                stage_score = self.calculate_stage_compatibility(
                    startup['funding_stage'], 
                    investor_stages
                )
                
                location_score = self.calculate_location_score(
                    startup['location'], 
                    investor_locations
                )
                
                growth_score = self.calculate_growth_score(
                    startup['growth_rate'], 
                    investor['min_growth_rate']
                )
                
                b2b_score = 1.0 if investor['b2b_focus'] is None else \
                           1.0 if startup['b2b'] == investor['b2b_focus'] else 0.5
                
                additional_factors[i, j] = np.mean([
                    stage_score,
                    location_score,
                    growth_score,
                    b2b_score
                ])
        
        return (base_similarity * 0.7) + (additional_factors * 0.3)
    
    def get_top_matches_for_startup(self, startup_id: int, top_n: int = 10) -> List[Dict]:
        
        startup_idx = self.startups_data[self.startups_data['id'] == startup_id].index[0]
        similarity_scores = self.similarity_matrix[startup_idx]
        
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]
        
        matches = []
        startup = self.startups_data.iloc[startup_idx]
        
        for idx in top_indices:
            investor = self.investors_data.iloc[idx]
            investor_stages = eval(investor['preferred_stages']) if isinstance(investor['preferred_stages'], str) else investor['preferred_stages']
            investor_locations = eval(investor['preferred_locations']) if isinstance(investor['preferred_locations'], str) else investor['preferred_locations']
            
            match_details = {
                'investor_id': investor['id'],
                'investor_name': investor['name'],
                'preferred_industries': investor['preferred_industry'],
                'similarity_score': round(similarity_scores[idx], 3),
                'stage_compatibility': self.calculate_stage_compatibility(
                    startup['funding_stage'], 
                    investor_stages
                ),
                'location_match': startup['location'] in investor_locations,
                'avg_check_size': investor['avg_check_size'],
                'investment_history': investor['investment_history']
            }
            matches.append(match_details)
        
        return matches
    
    def get_top_matches_for_investor(self, investor_id: int, top_n: int = 10) -> List[Dict]:
        
        investor_idx = self.investors_data[self.investors_data['id'] == investor_id].index[0]
        similarity_scores = self.similarity_matrix.T[investor_idx]
        
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]
        
        matches = []
        investor = self.investors_data.iloc[investor_idx]
        investor_stages = eval(investor['preferred_stages']) if isinstance(investor['preferred_stages'], str) else investor['preferred_stages']
        
        for idx in top_indices:
            startup = self.startups_data.iloc[idx]
            match_details = {
                'startup_id': startup['id'],
                'startup_name': startup['name'],
                'industry': startup['industry'],
                'similarity_score': round(similarity_scores[idx], 3),
                'valuation': startup['valuation'],
                'growth_rate': startup['growth_rate'],
                'team_size': startup['team_size'],
                'monthly_revenue': startup['monthly_revenue'],
                'funding_stage': startup['funding_stage'],
                'location': startup['location'],
                'stage_compatibility': self.calculate_stage_compatibility(
                    startup['funding_stage'], 
                    investor_stages
                )
            }
            matches.append(match_details)
        
        return matches
    
    def fit(self):
        
        self.preprocess_data()
        self.similarity_matrix = self.calculate_similarity()

def custom_recommendation():
    
    synthesizer = DataSynthesizer()
    startups_df = pd.read_csv('startups.csv')
    investors_df = pd.read_csv('investors.csv')
    
    # Custom weights emphasizing certain criteria
    custom_weights = {
        'industry': 0.4,      
        'valuation': 0.15,
        'profitability': 0.15,
        'team_size': 0.1,
        'equity': 0.1,
        'growth_rate': 0.1
    }
    
    # Initialize with custom weights
    recommender = StartupInvestorRecommender(startups_df, investors_df, custom_weights)
    recommender.fit()
    
    # Get recommendations
    startup_id = 1
    matches = recommender.get_top_matches_for_startup(startup_id, top_n=5)
    return matches

def add_new_startup_example():
    
    synthesizer = DataSynthesizer()
    startups_df = pd.read_csv('startups.csv')
    investors_df = pd.read_csv('investors.csv')
    
   
    new_startup = {
        'id': len(startups_df) + 1,
        'name': 'NewTechCo',
        'industry': 'AI/ML',
        'valuation': 5000000, 
        'profitability': 0.1, 
        'avg_price': 500,
        'team_size': 20,
        'equity_offered': 0.15,  
        'growth_rate': 0.8,     
        'founding_year': 2023,
        'monthly_revenue': 100000,
        'burn_rate': 50000,
        'funding_stage': 'Seed',
        'location': 'US',
        'b2b': True
    }
    
    startups_df = pd.concat([startups_df, pd.DataFrame([new_startup])], ignore_index=True)
    
    recommender = StartupInvestorRecommender(startups_df, investors_df)
    recommender.fit()
    
    print("\nGetting matches for new startup...")
    matches = recommender.get_top_matches_for_startup(new_startup['id'], top_n=5)
    return matches