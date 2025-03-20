import pandas as pd
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import time
import re
import pickle
from sklearn.impute import SimpleImputer
import optuna
import optunahub
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
    'stat/win-pct-all-games': 'Win%',
    'ranking/last-10-games-by-other': 'Last 10 Rating',
}

START_YEAR = 2007
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))
if 2020 in YEARS:
    YEARS.remove(2020)

CONV_DICT = {"Abilene Chr":"Abl Christian",
            "American Univ":"American",
            "Appalachian St":"App State",
            "Ark Little Rock":"Little Rock",
            "Ark Pine Bluff":"AR-Pine Bluff",
            "Boston Univ": "Boston U",
            "Col Charleston":"Charleston",
            "Central Conn":"C Connecticut",
            'Connecticut': 'UConn',
            'Detroit': 'Detroit Mercy',
            "ETSU":"E Tennessee St",
            "FL Atlantic":"Florida Atlantic",
            "FL Gulf Coast":"FGCU",
            "Gardner Webb":"Gardner-Webb",
            "Hawaii": "Hawai'i",
            "James Madison": "J Madison",
            "Kennesaw":"Kennesaw St",
            "Kent":"Kent St",
            "LIU Brooklyn": "LIU-Brooklyn",
            "Loyola-Chicago":"Loyola Chi",
            "MS Valley St":"Miss Valley St",
            "MTSU":"Middle Tenn",
            "Massachusetts":"UMass",
            "Miami FL":"Miami",
            "McNeese St":"McNeese",
            "North Florida":"N Florida",
            "North Texas": "N Texas",
            "Northern Iowa":"N Iowa",
            "Northwestern LA":"NW State",
            "Sam Houston St": "Sam Houston",
            "SUNY Albany":"Albany",
            "South Alabama":"S Alabama",
            "South Florida":"S Florida",
            "Southern Univ":"Southern",
            "St Bonavent":"St Bonaventure",
            "St Joseph's PA":"Saint Joseph's",
            "St Louis":"Saint Louis",
            "St Mary's CA":"Saint Mary's",
            "St Peter's":"Saint Peter's",
            "TAM C. Christi": "Texas A&M-CC",
            "TX Southern" :"Texas So",
            "UC Santa Barbara":"UCSB",
            "UNC Asheville": "NC Asheville",
            "UNC Greensboro":"NC Greensboro",
            "UNC Wilmington": "NC Wilmington",
            "UT San Antonio": "UTSA",
            "WI Green Bay":"Green Bay",
            "WI Milwaukee":"Milwaukee",
            "WKU": "W Kentucky",
            }


# ======================
# IMPROVED WEB SCRAPER
# ======================
def scrape_team_rankings(year, stat_url):
    """Scrape team statistics and maintain year column"""
    url = f"https://www.teamrankings.com/ncaa-basketball/{stat_url}?date={year}-03-18"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        html = BeautifulSoup(response.content, 'html.parser')
        html_io = StringIO(str(html))
        df = pd.read_html(html_io)[0].iloc[:, 1:3]
        df.columns = ['Team', 'Value']
        
        # Clean team names
        df['Team'] = (
            df['Team']
            .str.replace(r'\s*\(\d+.*?\)', '', regex=True)
            .str.strip()
        )
        
        # Convert values
        df['Value'] = (
            df['Value']
            .astype(str)
            .str.replace('%', '')
            .apply(pd.to_numeric, errors='coerce')
        )
        
        if '%' in stat_url:
            df['Value'] /= 100
            
        df['Year'] = year
        df['Stat'] = STATS[stat_url]
        
        return df[['Year', 'Team', 'Stat', 'Value']].dropna()
        
    except Exception as e:
        print(f"Error scraping {stat_url} for {year}: {str(e)}")
        return pd.DataFrame()

# ======================
# DATA PROCESSING
# ======================
def process_features(matchups):
    """Main data processing pipeline with cached scraping"""
    # Load or scrape data
    csv_path = 'team_stats.csv'
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing_years = existing['Year'].unique()
        missing_years = [y for y in YEARS if y not in existing_years]
        
        if missing_years:
            print(f"Scraping missing years: {missing_years}")
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for year in missing_years:
                    for stat_url in STATS.keys():
                        futures.append(executor.submit(scrape_team_rankings, year, stat_url))
                        time.sleep(0.5)
                
                new_data = pd.concat([f.result() for f in futures if not f.result().empty])
                combined = pd.concat([existing, new_data])
                combined.to_csv(csv_path, index=False)
                raw_stats = combined
        else:
            raw_stats = existing
    else:
        print("Scraping all data...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for year in YEARS:
                for stat_url in STATS.keys():
                    futures.append(executor.submit(scrape_team_rankings, year, stat_url))
                    time.sleep(0.5)
            
            raw_stats = pd.concat([f.result() for f in futures if not f.result().empty])
            raw_stats.to_csv(csv_path, index=False)
    
    # Convert team names using dictionary
    matchups['Team1'] = matchups['Team1'].map(CONV_DICT).fillna(matchups['Team1'])
    matchups['Team2'] = matchups['Team2'].map(CONV_DICT).fillna(matchups['Team2'])
    
    # Create pivot table
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
            t1_stats = team_stats[(team_stats['Year'] == year) & (team_stats['Team'] == t1)].iloc[0]
            t2_stats = team_stats[(team_stats['Year'] == year) & (team_stats['Team'] == t2)].iloc[0]
            
            feature_row = {'Year': year, 'Team1': t1, 'Team2': t2}
            for stat in STATS.values():
                # Get base values
                t1_val = t1_stats[stat]
                t2_val = t2_stats[stat]
                
                # Calculate momentum weights
                t1_momentum = 1 + (t1_stats['Last 10 Rating'] * 0.5)  # 50% weight to recent form
                t2_momentum = 1 + (t2_stats['Last 10 Rating'] * 0.5)
                
                # Calculate SOS weights
                t1_sos = 1 + (t1_stats['SOS'] * 0.3)  # 30% weight to schedule strength
                t2_sos = 1 + (t2_stats['SOS'] * 0.3)
                
                # Apply weighted values
                t1_weighted = t1_val * t1_momentum * t1_sos
                t2_weighted = t2_val * t2_momentum * t2_sos
                
                # Create features
                feature_row[f'{stat}_Diff'] = t1_weighted - t2_weighted
                feature_row[f'{stat}_Ratio'] = t1_weighted / (t2_weighted + 1e-8)
                feature_row[f'{stat}_Product'] = t1_weighted * t2_weighted
                feature_row[f'{stat}_SquaredDiff'] = (t1_weighted - t2_weighted)**2
                
            feature_row['Target'] = row['Target']
            features.append(feature_row)
            
        except IndexError:
            print(f"Missing data for {year}: {t1} vs {t2}")
            continue
    
    # Post-processing
    feature_df = pd.DataFrame(features)
    feature_cols = [c for c in feature_df.columns if c not in ['Year', 'Team1', 'Team2', 'Target']]
    
    feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df[feature_cols] = SimpleImputer(strategy='median').fit_transform(feature_df[feature_cols])
    
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
    

    def objective(trial):
        # Define hyperparameter space for base models
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
            # 'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            # 'criterion': trial.suggest_categorical('rf_criterion', ['log_loss'])
        }

        gb_params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('gb_max_depth', 1, 10)
        }

        lr_params = {
            'C': trial.suggest_float('lr_C', 0.01, 1.0, log=True),
            'max_iter': trial.suggest_int('lr_max_iter', 50, 200),
            # 'penalty': trial.suggest_categorical('lr_penalty', ['l2']),
            # 'solver': trial.suggest_categorical('lr_solver', ['saga'])
        }

        # cnb_params = {
        #     'alpha': trial.suggest_float('cnb_alpha', 0.01, 0.3, log=True),
        #     'norm': trial.suggest_categorical('cnb_norm', [True])
        # }

        # mlp_params = {
        #     'hidden_layer_sizes': tuple(
        #     [trial.suggest_int('mlp_units', 50, 200)] 
        #     * trial.suggest_int('mlp_layers', 1, 10)
        # ),
        #     'activation': trial.suggest_categorical('mlp_activation', ['relu']),
        #     'alpha': trial.suggest_float('mlp_alpha', 1e-4, 1e-1, log=True),
        #     'learning_rate_init': trial.suggest_float('mlp_lr_init', 1e-4, 0.1, log=True),
        #     'max_iter': trial.suggest_int('mlp_max_iter', 500, 2000),
        #     'tol': trial.suggest_float('mlp_tol', 1e-5, 1e-3, log=True)
        # }

        # Create base estimators
        estimators = [
            ('rf', RandomForestClassifier(**rf_params)),
            ('gb', GradientBoostingClassifier(**gb_params)),
            ('lr', LogisticRegression(**lr_params)),
            # ('cnb', ComplementNB(**cnb_params)),
            # ('mlp', MLPClassifier(**mlp_params))
        ]

        # Create voting classifier
        vote = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities for better log loss
            n_jobs=-1
        )

        # Cross-validated evaluation
        score = cross_val_score(
            vote,
            X_train_scaled,
            y_train,
            scoring='neg_log_loss',
            n_jobs=-1
        ).mean()

        return -score


    # Optimize with Optuna
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    # Train final model
    best_params = study.best_params
    rf_best = {k.split('_',1)[1]: v for k,v in best_params.items() if k.startswith('rf_')}
    gb_best = {k.split('_',1)[1]: v for k,v in best_params.items() if k.startswith('gb_')}
    lr_best = {k.split('_',1)[1]: v for k,v in best_params.items() if k.startswith('lr_')}
    # cnb_best = {k.split('_',1)[1]: v for k,v in best_params.items() if k.startswith('cnb_')}
    
    # mlp_best = {
    #     'hidden_layer_sizes': tuple([best_params['mlp_units']] * best_params['mlp_layers']),
    #     'activation': best_params['mlp_activation'],
    #     'alpha': best_params['mlp_alpha'],
    #     'learning_rate_init': best_params['mlp_lr_init'],
    #     'max_iter': best_params['mlp_max_iter'],
    #     'tol': best_params['mlp_tol']
    # }

    final_vote = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(**rf_best)),
            ('gb', GradientBoostingClassifier(**gb_best)),
            ('lr', LogisticRegression(**lr_best)),
            # ('cnb', ComplementNB(**cnb_best)),
            # ('mlp', MLPClassifier(**mlp_best))
        ],
        voting='soft',
        n_jobs=-1
    )

    final_vote.fit(X_train_scaled, y_train)

    # Evaluate
    probs = final_vote.predict_proba(X_test_scaled)[:, 1]
    loss = log_loss(y_test, probs)
    accuracy = accuracy_score(y_test, final_vote.predict(X_test_scaled))

    print(f"\nBest Voting Classifier - Loss: {loss:.4f}, Accuracy: {accuracy:.3f}")

    # Save artifacts
    artifacts = {
        'model': final_vote,
        'scaler': scaler,
        'features': feature_cols,
        'stats_config': STATS,
        'best_params': best_params
    }

    with open('voting_model.pkl', 'wb') as f:
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
    
    print("\nTraining complete. Best models saved.")
    # for name, data in model_artifacts['models'].items():
    #     print(f"- {name} (Log Loss: {data['logloss']:.3f})")
    # print(f"Overall best model: {model_artifacts['best_model']}")
