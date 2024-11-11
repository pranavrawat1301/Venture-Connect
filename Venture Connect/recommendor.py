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
        
        self.feature_weights = feature_weights or {
            'industry': 0.2,  # Reduced from 0.3
            'valuation': 0.2,
            'profitability': 0.2,
            'team_size': 0.1,
            'equity': 0.15,
            'growth_rate': 0.15
        }
        
        self.industry_groups = {
            'tech': ['AI/ML', 'SaaS', 'Cloud Computing', 'Cybersecurity', 'IoT'],
            'finance': ['FinTech', 'InsurTech', 'Blockchain', 'Cryptocurrency'],
            'health': ['HealthTech', 'BioTech', 'MedTech', 'Digital Health'],
            'consumer': ['E-commerce', 'Consumer Apps', 'Retail Tech', 'MarketPlace'],
            'enterprise': ['Enterprise Software', 'B2B Services', 'Data Analytics']
        }
        
        self.industry_similarity_matrix = self._create_industry_similarity_matrix()
        self.startup_scaler = MinMaxScaler()
        self.investor_scaler = MinMaxScaler()
        
        self.preprocess_data()
        self.similarity_matrix = self.calculate_similarity()
    
    def _create_industry_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        all_industries = set()
        for group in self.industry_groups.values():
            all_industries.update(group)
        
        similarity_matrix = {}
        for ind1 in all_industries:
            similarity_matrix[ind1] = {}
            for ind2 in all_industries:
                if ind1 == ind2:
                    similarity_matrix[ind1][ind2] = 1.0
                else:
                    same_group = False
                    for group in self.industry_groups.values():
                        if ind1 in group and ind2 in group:
                            same_group = True
                            break
                    similarity_matrix[ind1][ind2] = 0.5 if same_group else 0.1
        return similarity_matrix
    
    def _calculate_industry_similarity(self, startup_industry: str, investor_industries: List[str]) -> float:
        if not isinstance(investor_industries, list):
            investor_industries = eval(investor_industries)
        
        similarities = []
        for inv_industry in investor_industries:
            try:
                similarity = self.industry_similarity_matrix[startup_industry][inv_industry]
                similarities.append(similarity)
            except KeyError:
                similarities.append(0.1)  
        
        return max(similarities) if similarities else 0.1
    
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
        startup_numerical = self.startups_data[['valuation', 'profitability', 
                                              'team_size', 'equity_offered', 'growth_rate']]
        startup_numerical_scaled = self.startup_scaler.fit_transform(startup_numerical)
        
        investor_numerical = self.investors_data[['max_valuation', 'target_profitability', 
                                                'preferred_team_size', 'target_equity', 'min_growth_rate']]
        investor_numerical_scaled = self.investor_scaler.fit_transform(investor_numerical)
        
        industry_similarity_matrix = np.zeros((len(self.startups_data), len(self.investors_data)))
        for i, startup in self.startups_data.iterrows():
            for j, investor in self.investors_data.iterrows():
                industry_similarity_matrix[i, j] = self._calculate_industry_similarity(
                    startup['industry'],
                    investor['preferred_industry']
                )
        
        self.numerical_similarity = cosine_similarity(startup_numerical_scaled, investor_numerical_scaled)
        self.industry_similarity = industry_similarity_matrix
    
    def calculate_similarity(self) -> np.ndarray:
        numerical_weight = 1 - self.feature_weights['industry']
        industry_weight = self.feature_weights['industry']
        
        final_similarity = (numerical_weight * self.numerical_similarity + 
                          industry_weight * self.industry_similarity)
        
        additional_factors = np.zeros_like(final_similarity)
        
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
        
        return (final_similarity * 0.7) + (additional_factors * 0.3)
    
    def get_top_matches_for_startup(self, startup_id: int, top_n: int = 10) -> str:
        startup_idx = self.startups_data[self.startups_data['id'] == startup_id].index[0]
        similarity_scores = self.similarity_matrix[startup_idx]
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]

        startup = self.startups_data.iloc[startup_idx]
        matches = []

        for idx in top_indices:
            investor = self.investors_data.iloc[idx]
            score = similarity_scores[idx] * 5
            similarity = self._calculate_industry_similarity(startup['industry'], investor['preferred_industry'])
            
            matches.append(f"Investor ID: {investor['id']:<5} | "
                         f"Name: {investor['name']:<20} | "
                         f"Industry Match: {', '.join(eval(investor['preferred_industry']) if isinstance(investor['preferred_industry'], str) else investor['preferred_industry']):<30} | "
                         f"Score: {score:.2f}/5")

        result = f"Top {top_n} Matches for Startup ID: {startup_id} (Industry: {startup['industry']})\n" + "\n".join(matches)
        return result
    
    def get_top_matches_for_investor(self, investor_id: int, top_n: int = 10) -> str:
        investor_idx = self.investors_data[self.investors_data['id'] == investor_id].index[0]
        similarity_scores = self.similarity_matrix.T[investor_idx]
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]

        investor = self.investors_data.iloc[investor_idx]
        matches = []

        for idx in top_indices:
            startup = self.startups_data.iloc[idx]
            score = similarity_scores[idx] * 5
            similarity = self._calculate_industry_similarity(startup['industry'], investor['preferred_industry'])
            
            matches.append(f"Startup ID: {startup['id']:<5} | "
                         f"Name: {startup['name']:<20} | "
                         f"Industry: {startup['industry']:<20} | "
                         f"Score: {score:.2f}/5")

        result = f"Top {top_n} Matches for Investor ID: {investor_id}\n" + "\n".join(matches)
        return result
    
    def fit(self):
        self.preprocess_data()
        self.similarity_matrix = self.calculate_similarity()

def custom_recommendation():
    synthesizer = DataSynthesizer()
    startups_df = pd.read_csv('startups.csv')
    investors_df = pd.read_csv('investors.csv')
    
    custom_weights = {
        'industry': 0.2,      
        'valuation': 0.2,
        'profitability': 0.2,
        'team_size': 0.1,
        'equity': 0.15,
        'growth_rate': 0.15
    }
    
    recommender = StartupInvestorRecommender(startups_df, investors_df, custom_weights)
    
    startup_id = 1
    matches = recommender.get_top_matches_for_startup(startup_id, top_n=5)
    return matches

def add_new_startup():
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
    
    print("\nGetting matches for new startup...")
    matches = recommender.get_top_matches_for_startup(new_startup['id'], top_n=5)
    return matches