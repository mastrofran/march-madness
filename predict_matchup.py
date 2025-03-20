# predict_matchup.py
import pickle
import argparse
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
from thefuzz import process, fuzz
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class MarchMadnessPredictor:
    def __init__(self, model_path):
        """Load trained stacking model"""
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.feature_names = artifacts['features']
        self.stats_config = artifacts['stats_config']
        self.team_stats_cache = {}

    def predict_matchups(self, matchups_df, year):
        """Predict multiple matchups from a dataframe"""
        # Get unique teams
        all_teams = pd.unique(matchups_df[['Team1', 'Team2']].values.ravel('K'))
        
        # Scrape all stats at once
        self._scrape_all_team_stats(year)
        
        # Process all matchups
        results = []
        for _, row in tqdm(matchups_df.iterrows(), total=len(matchups_df)):
            prediction = self._predict_single_matchup(row['Team1'], row['Team2'])
            results.append(prediction)
            
        return pd.DataFrame(results)
    
    def _scrape_all_team_stats(self, year):
        """Scrape all team stats from each stat table"""
        print(f"\nScraping stats for {year}...")
        
        for stat_url, stat_key in tqdm(self.stats_config.items(), desc="Stats"):
            try:
                url = f"https://www.teamrankings.com/ncaa-basketball/{stat_url}?date={year}-03-18"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                
                # Parse table
                html = BeautifulSoup(response.content, 'html.parser')
                html_io = StringIO(str(html))
                df = pd.read_html(html_io)[0].iloc[:, 1:3]
                df.columns = ['Team', stat_key]
                
                # Clean team names
                df['Team'] = df['Team'].str.replace(r'\s*\(\d+.*?\)', '', regex=True).str.strip()
                
                # Store stats in cache
                for _, row in df.iterrows():
                    team = row['Team']
                    value = row[stat_key]
                    
                    if team not in self.team_stats_cache:
                        self.team_stats_cache[team] = {}
                    
                    # Convert percentage values
                    if '%' in str(value):
                        self.team_stats_cache[team][stat_key] = float(value.replace('%', '')) / 100
                    else:
                        self.team_stats_cache[team][stat_key] = float(value)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error scraping {stat_key}: {str(e)}")
                continue

    def _create_features(self, team1_stats, team2_stats):
        """Create feature vector from cached stats"""
        features = {}
        
        for stat in self.stats_config.values():
            val1 = team1_stats.get(stat, 0)
            val2 = team2_stats.get(stat, 0)
            
            t1_momentum = 1 + (team1_stats['Last 10 Rating'] * 0.3)  # 30% weight to recent form
            t2_momentum = 1 + (team2_stats['Last 10 Rating'] * 0.3)
            
            # Calculate SOS weights
            t1_sos = 1 + (team1_stats['SOS'] * 0.3)  # 30% weight to schedule strength
            t2_sos = 1 + (team2_stats['SOS'] * 0.3)
            
            # Apply weighted values
            t1_weighted = val1 * t1_momentum * t1_sos
            t2_weighted = val2 * t2_momentum * t2_sos
            
            # Create features
            features[f'{stat}_Diff'] = t1_weighted - t2_weighted
            features[f'{stat}_Ratio'] = t1_weighted / (t2_weighted + 1e-8)
            features[f'{stat}_Product'] = t1_weighted * t2_weighted
            features[f'{stat}_SquaredDiff'] = (t1_weighted - t2_weighted)**2
        
        return pd.DataFrame([features])[self.feature_names]
    
    def _predict_single_matchup(self, team1, team2):
        """Get predictions from all models for a single matchup"""
        # Find best matches for team names
        all_teams = list(self.team_stats_cache.keys())
        if team1 not in all_teams:
            team1_match = process.extractOne(team1, all_teams, scorer=fuzz.token_set_ratio)[0]
            # raise ValueError(f"Team {team1} not found in stats cache. Closest match: {team1_match}")
            print(f"Team {team1} not found in stats cache. Closest match: {team1_match}")
        else:
            team1_match = team1
        
        if team2 not in all_teams:
            team2_match = process.extractOne(team2, all_teams, scorer=fuzz.token_set_ratio)[0]
            # raise ValueError(f"Team {team2} not found in stats cache. Closest match: {team2_match}")
            print(f"Team {team2} not found in stats cache. Closest match: {team2_match}")

        else:
            team2_match = team2
        # team1_match = process.extractOne(team1, all_teams, scorer=fuzz.token_set_ratio)[0]
        # team2_match = process.extractOne(team2, all_teams, scorer=fuzz.token_set_ratio)[0]
        
        # Get stats for matched teams
        team1_stats = self.team_stats_cache[team1_match]
        team2_stats = self.team_stats_cache[team2_match]
        
        # Create features and predict
        features = self._create_features(team1_stats, team2_stats)
        features_scaled = self.scaler.transform(features)
        
         # Get prediction and probability
        prob = self.model.predict_proba(features_scaled)[0][1]
        # Get individual model probabilities for interpretability
        model_probs = {}
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'predict_proba'):
                model_probs[name] = estimator.predict_proba(features_scaled)[0][1]
            else:
                decision = estimator.decision_function(features_scaled)
                model_probs[name] = 1 / (1 + np.exp(-decision))[0]
        
        return {
            'Team1': team1,
            'Team2': team2,
            'Winner': team1 if prob > 0.5 else team2,
            'Probability': f"{max(prob, 1-prob):.1%}",
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble March Madness Predictor')
    parser.add_argument('input_csv', help='CSV with Team1,Team2 columns')
    parser.add_argument('output_csv', help='Output file path')
    parser.add_argument('--year', type=int, default=2024, help='Prediction year')
    args = parser.parse_args()

    # Load matchups
    matchups = pd.read_csv(args.input_csv)
    
    predictor = MarchMadnessPredictor('voting_model.pkl')
    results = predictor.predict_matchups(matchups, args.year)
     # Clean output
    results = results[['Team1', 'Team2', 'Winner', 'Probability',]]
    results.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")