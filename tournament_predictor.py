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
import pickle
from sklearn.impute import SimpleImputer
import hashlib
from joblib import Memory
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

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
# MEMORY CACHE
# ======================
memory = Memory(location='./cache', verbose=0)

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
@memory.cache
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

            # New: Interaction terms
            feature_row[f'{stat}_Product'] = t1_stats[stat] * t2_stats[stat]
            feature_row[f'{stat}_SquaredDiff'] = (t1_stats[stat] - t2_stats[stat])**2
            
            feature_row['Target'] = row['Target']
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
    """Train and evaluate predictive models with Optuna optimization"""
    feature_cols = [col for col in feature_df.columns 
                   if '_Diff' in col or '_Ratio' in col or '_Product' in col or '_SquaredDiff' in col]
    X = feature_df[feature_cols]
    y = features['Target']

    # Handle missing values
    if X.isna().sum().sum() > 0:
        print("Warning: Data still contains missing values after imputation")
        print(X.isna().sum())
        X = X.dropna()
        y = y.loc[X.index]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_models = {}
    
    # Define model configurations with Optuna search spaces
    model_configs = {
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': lambda trial: {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 2000),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'params': lambda trial: {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
        },
        'Gradient Boosting': {
            'class': GradientBoostingClassifier,
            'params': lambda trial: {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        },
        'Linear SVC': {
            'class': LinearSVC,
            'params': lambda trial: {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'dual': trial.suggest_categorical('dual', [False]),
                'max_iter': trial.suggest_int('max_iter', 1000, 20000)
            },
            'calibrated': True
        }
    }

    for model_name, config in model_configs.items():
        print(f"\nOptimizing {model_name}...")

        def objective(trial):
            params = config['params'](trial)
            
            if config.get('calibrated', False):
                base_model = config['class'](**params)
                model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            else:
                model = config['class'](**params)
                
            score = cross_val_score(
                model, 
                X_train_scaled, 
                y_train, 
                cv=3, 
                scoring='neg_log_loss',
                n_jobs=-1
            ).mean()
            return -score  # Minimize negative log loss


        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        
        # Train final model with best params
        best_params = study.best_params
        if config.get('calibrated', False):
            base_model = config['class'](**best_params)
            final_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        else:
            final_model = config['class'](**best_params)
            
        final_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        if hasattr(final_model, 'predict_proba'):
            probs = final_model.predict_proba(X_test_scaled)[:, 1]
        else:
            decision = final_model.decision_function(X_test_scaled)
            probs = 1 / (1 + np.exp(-decision))
            
        logloss = log_loss(y_test, probs)
        accuracy = accuracy_score(y_test, final_model.predict(X_test_scaled))
        
        best_models[model_name] = {
            'model': final_model,
            'logloss': logloss,
            'accuracy': accuracy,
            'params': best_params
        }
        
        print(f"{model_name} - Best Log Loss: {logloss:.4f}, Accuracy: {accuracy:.3f}")

    # Determine best overall model
    best_overall = min(best_models.items(), key=lambda x: x[1]['logloss'])
    
    # Save artifacts
    artifacts = {
        'models': {name: data['model'] for name, data in best_models.items()},
        'best_model': best_overall[0],
        'best_params': best_overall[1]['params'],
        'scaler': scaler,
        'features': feature_cols,
        'stats_config': STATS
    }

    with open('tournament_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    return artifacts


def create_balanced_matchups(results_df, teams_df):
        """Create matchups with randomized team order and proper labeling"""
        merged = results_df.merge(
            teams_df[['TeamID', 'TeamName']],
            left_on='WTeamID',
            right_on='TeamID'
        ).merge(
            teams_df[['TeamID', 'TeamName']],
            left_on='LTeamID',
            right_on='TeamID',
            suffixes=('_W', '_L')
        )
        
        # Randomize team order
        np.random.seed(42)
        swap_mask = np.random.rand(len(merged)) < 0.5
        
        return pd.DataFrame({
            'Year': merged['Season'],
            'Team1': np.where(swap_mask, merged['TeamName_L'], merged['TeamName_W']),
            'Team2': np.where(swap_mask, merged['TeamName_W'], merged['TeamName_L']),
            'Target': np.where(swap_mask, 0, 1)  # 1 if Team1 is actual winner
        })

# def objective(trial):
#         params = config['params'](trial)
        
#         if config.get('calibrated', False):
#             base_model = config['class'](**params)
#             model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
#         else:
#             model = config['class'](**params)
            
#         score = cross_val_score(
#             model, 
#             X_train_scaled, 
#             y_train, 
#             cv=3, 
#             scoring='neg_log_loss',
#             n_jobs=-1
#         ).mean()
#         return -score  # Minimize negative log loss

# ======================
# MAIN WORKFLOW
# ======================
if __name__ == "__main__":
    # Load historical matchups
    teams = pd.read_csv("data/MTeams.csv")
    results = pd.read_csv("data/MNCAATourneyCompactResults.csv")

    results = results[results['Season'].isin(YEARS)]
    
    matchups = create_balanced_matchups(results, teams)
    
    # Process features with fuzzy matching
    feature_df = process_features(matchups)
    
    # Train and evaluate models
    print("\nStarting model training...")
    model_artifacts = train_model(feature_df)
    
    print("\nTraining complete. Best models saved:")
    # for name, data in model_artifacts['models'].items():
    #     print(f"- {name} (Log Loss: {data['logloss']:.3f})")
    print(f"Overall best model: {model_artifacts['best_model']}")
