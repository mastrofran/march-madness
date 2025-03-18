import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from concurrent.futures import ThreadPoolExecutor
from thefuzz import process, fuzz
from io import StringIO
import time
import re
from xgboost import XGBClassifier
import pickle
from sklearn.impute import SimpleImputer

# ======================
# CONFIGURATION
# ======================
STATS = {
    'stat/offensive-efficiency': 'OffEff',
    'stat/defensive-efficiency': 'DefEff',
    'stat/effective-field-goal-pct': 'eFG%',
    'stat/opponent-effective-field-goal-pct': 'Def_eFG%',
    'stat/turnovers-per-possession': 'TOV%',
    'stat/offensive-rebounding-pct': 'ORB%',
    'stat/defensive-rebounding-pct': 'DRB%',
    'stat/free-throw-rate': 'FTRate',
    'stat/three-point-pct': '3P%',
    'stat/steal-pct': 'STL%',
    'stat/block-pct': 'BLK%',
    'ranking/schedule-strength-by-other': 'SOS',
    'stat/true-shooting-percentage': 'TS%',
    'stat/possessions-per-game': 'Pace',
    'stat/win-pct-close-games': 'CloseWin%',
    'ranking/last-10-games-by-other': 'Last 10 Rating',
}

START_YEAR = 2007
END_YEAR = 2025
YEARS = list(range(START_YEAR, END_YEAR + 1))
if 2020 in YEARS:
    YEARS.remove(2020)

# ======================
# FUZZY MATCHING CLASS
# ======================
class FuzzyTeamMatcher:
    def __init__(self, target_names):
        self.target_names = target_names
        self.cache = {}
        self.unmatched = set()
        
    def find_best_match(self, name, threshold=85):
        if name in self.cache:
            return self.cache[name]
            
        # Use token set ratio for better matching
        matches = process.extractBests(
            name, 
            self.target_names, 
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=3
        )
        
        if not matches:
            self.unmatched.add(name)
            return name
            
        best_match = max(matches, key=lambda x: (x[1], -len(x[0])))
        self.cache[name] = best_match[0]
        return best_match[0]

# ======================
# IMPROVED WEB SCRAPER
# ======================
def scrape_team_rankings(year, stat_url):
    """Scrape team statistics using pandas read_html with fuzzy cleaning"""
    url = f"https://www.teamrankings.com/ncaa-basketball/{stat_url}?date={year}-03-18"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Use StringIO to avoid pandas warning
        html = BeautifulSoup(response.content, 'html.parser')
        html_io = StringIO(str(html))
        
        tables = pd.read_html(html_io)
        if not tables:
            return pd.DataFrame()
            
        table = tables[0].iloc[:, 1:3]
        table.columns = ['Team', 'Value']
        
        # Clean team names
        table['Team'] = (
            table['Team']
            .astype(str)
            .str.replace(r'\s*\(\d+\)', '', regex=True)  # Remove rankings
            .str.replace(r'\s*\(\d+-\d+\)', '', regex=True)  # Remove records
            .str.strip()
        )
        
        # Convert values
        table['Value'] = (
            table['Value']
            .astype(str)
            .str.replace('%', '')
            .apply(pd.to_numeric, errors='coerce')
        )
        
        if '%' in stat_url:
            table['Value'] /= 100
            
        table['Year'] = year
        table['Stat'] = STATS[stat_url]
        
        return table[['Year', 'Team', 'Stat', 'Value']].dropna()
        
    except Exception as e:
        print(f"Error scraping {stat_url} for {year}: {str(e)}")
        return pd.DataFrame()

# ======================
# DATA PROCESSING
# ======================
def process_features(matchups):
    """Main data processing pipeline with fuzzy matching"""
    # Scrape all team stats
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for year in YEARS:
            for stat_url in STATS.keys():
                futures.append(executor.submit(scrape_team_rankings, year, stat_url))
                time.sleep(0.5)
            print(f"Scraped stats for {year}")
        
        dfs = [f.result() for f in futures if not f.result().empty]
        raw_stats = pd.concat(dfs)
    
    # Get unique scraped team names for matching
    scraped_teams = raw_stats['Team'].unique().tolist()
    
    # Initialize fuzzy matcher
    matcher = FuzzyTeamMatcher(scraped_teams)
    
    # Align matchup team names
    matchups['Team1'] = matchups['Team1'].apply(matcher.find_best_match)
    matchups['Team2'] = matchups['Team2'].apply(matcher.find_best_match)
    
    # Create pivot table of stats
    team_stats = raw_stats.pivot_table(
        index=['Year', 'Team'],
        columns='Stat',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    # Create features
    features = []
    for _, row in matchups.iterrows():
        year = row['Year']
        t1 = row['Team1']
        t2 = row['Team2']
        
        try:
            t1_stats = team_stats[(team_stats['Year'] == year) & 
                                 (team_stats['Team'] == t1)].iloc[0]
            t2_stats = team_stats[(team_stats['Year'] == year) & 
                                 (team_stats['Team'] == t2)].iloc[0]
            
            feature_row = {'Year': year, 'Team1': t1, 'Team2': t2}
            for stat in STATS.values():
                feature_row[f'{stat}_Diff'] = t1_stats[stat] - t2_stats[stat]
                feature_row[f'{stat}_Ratio'] = t1_stats[stat] / (t2_stats[stat] + 1e-8)
            
            feature_row['Target'] = 1 if row['Winner'] == t1 else 0
            features.append(feature_row)
            
        except IndexError:
            print(f"Missing data for {year}: {t1} vs {t2}")
            continue
    
    # Report unmatched teams
    if matcher.unmatched:
        print("\nTeams needing manual review:")
        for team in sorted(matcher.unmatched):
            print(f"- {team}")
    
     # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
     # Separate metadata from numeric features
    meta_cols = ['Year', 'Team1', 'Team2']
    feature_cols = [col for col in feature_df.columns if col not in meta_cols]
    
    # Handle infinite values
    feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Impute only numeric features
    imputer = SimpleImputer(strategy='median')
    feature_df[feature_cols] = imputer.fit_transform(feature_df[feature_cols])
    
    return feature_df

# ======================
# MODEL TRAINING
# ======================
def train_model(features):
    """Train and evaluate predictive models"""
    feature_cols = [col for col in feature_df.columns 
                   if '_Diff' in col or '_Ratio' in col]
    X = feature_df[feature_cols]
    y = features['Target']

    # +++ ADD THIS VALIDATION +++
    # Final check for remaining missing values
    if X.isna().sum().sum() > 0:
        print("Warning: Data still contains missing values after imputation")
        print(X.isna().sum())
        X = X.dropna()
        y = y.loc[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(
            penalty='l2', 
            C=0.1, 
            solver='liblinear',
            max_iter=2000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            eval_metric='logloss'
        ),
        'Linear SVC': LinearSVC(
            dual=False,
            max_iter=5000,
            class_weight='balanced'
        )
    }
    
    best_models = {}
    best_overall = {'name': None, 'score': float('inf')}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train with early stopping
        if name == 'XGBoost':
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)
        
        # Track best model based on log loss
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            decision = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-decision))
            
        # Calculate metrics
        current_logloss = log_loss(y_test, probs)
        current_accuracy = accuracy_score(y_test, model.predict(X_test))

        # Track best model version
        if name not in best_models or current_logloss < best_models[name]['logloss']:
            best_models[name] = {
                'model': model,
                'logloss': current_logloss,
                'accuracy': current_accuracy
            }

        # Track overall best model
        if current_logloss < best_overall['score']:
            best_overall['name'] = name
            best_overall['score'] = current_logloss

        print(f"{name} - Log Loss: {current_logloss:.3f}, Accuracy: {current_accuracy:.3f}")
    
    # Save all best models
    artifacts = {
        'models': {name: data['model'] for name, data in best_models.items()},
        'best_model': best_overall['name'],
        'scaler': scaler,
        'features': feature_cols,
        'stats_config': STATS
    }

    with open('tournament_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    return artifacts

# ======================
# MAIN WORKFLOW
# ======================
if __name__ == "__main__":
    # Load historical matchups
    teams = pd.read_csv("data/MTeams.csv")
    results = pd.read_csv("data/MNCAATourneyCompactResults.csv")
    
    # Create base matchups
    matchups = results.merge(
        teams[['TeamID', 'TeamName']],
        left_on='WTeamID',
        right_on='TeamID'
    ).merge(
        teams[['TeamID', 'TeamName']],
        left_on='LTeamID',
        right_on='TeamID',
        suffixes=('_1', '_2')
    ).rename(columns={
        'Season': 'Year',
        'TeamName_1': 'Team1',
        'TeamName_2': 'Team2'
    })[['Year', 'Team1', 'Team2']]
    
    matchups['Winner'] = matchups['Team1']
    matchups = matchups[matchups['Year'].isin(YEARS)]
    
    # Process features with fuzzy matching
    feature_df = process_features(matchups)
    
    # Train and evaluate models
    print("\nStarting model training...")
    model_artifacts = train_model(feature_df)
    
    print("\nTraining complete. Best models saved:")
    # for name, data in model_artifacts['models'].items():
    #     print(f"- {name} (Log Loss: {data['logloss']:.3f})")
    print(f"Overall best model: {model_artifacts['best_model']}")
