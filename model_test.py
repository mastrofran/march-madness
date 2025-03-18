# model_test.py
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, log_loss
from predict_matchup import MarchMadnessPredictor  # Assuming EnsemblePredictor is in separate file

def load_and_preprocess_data(results_path, teams_path, start_year=2007):
    """Load and prepare historical tournament data"""
    # Load data
    results = pd.read_csv(results_path)
    teams = pd.read_csv(teams_path)[['TeamID', 'TeamName']]
    
    # Filter to modern era
    results = results[results['Season'] >= start_year]
    
    # Convert team IDs to names
    results = results.merge(
        teams, 
        left_on='WTeamID',
        right_on='TeamID'
    ).merge(
        teams,
        left_on='LTeamID',
        right_on='TeamID',
        suffixes=('_W', '_L')
    )
    
    # Create matchup format
    matchups = pd.DataFrame({
        'Season': results['Season'],
        'Team1': results['TeamName_W'],
        'Team2': results['TeamName_L'],
        'ActualWinner': results['TeamName_W']  # Since WTeam always won in historical data
    })

    print(matchups.head())
    
    return matchups

def evaluate_model(model_path, test_data):
    """Evaluate model on historical matchups"""
    # Initialize predictor
    predictor = MarchMadnessPredictor(model_path)
    
    # Prepare containers for metrics
    metrics = {
        'models': {},
        'ensemble': {'true': [], 'pred': [], 'probs': []},
        'total': 0,
        'correct': 0
    }
    
    # Process by season to ensure proper stat scraping
    for year, year_data in test_data.groupby('Season'):
        print(f"\nEvaluating {year} season...")
        
        # Get predictions for all matchups in this year
        try:
            preds = predictor.predict_matchups(year_data[['Team1', 'Team2']], year)
            merged = year_data.merge(preds, on=['Team1', 'Team2'])
            
            # Collect metrics for each model
            for model in predictor.models.keys():
                model_col = f'{model}_Winner'
                actual = merged['ActualWinner'].values
                pred = merged[model_col].values
                
                acc = accuracy_score(actual, pred)
                if model not in metrics['models']:
                    metrics['models'][model] = {'acc': [], 'total': 0}
                
                metrics['models'][model]['acc'].append(acc)
                metrics['models'][model]['total'] += len(merged)
            
            # Collect ensemble metrics
            metrics['ensemble']['true'].extend(merged['ActualWinner'])
            metrics['ensemble']['pred'].extend(merged['Final_Prediction'])
            metrics['correct'] += (merged['Final_Prediction'] == merged['ActualWinner']).sum()
            metrics['total'] += len(merged)
            
        except Exception as e:
            print(f"Error evaluating {year}: {str(e)}")
            continue
    
    # Calculate final metrics
    results = {}
    
    # Individual model performance
    for model, data in metrics['models'].items():
        weighted_acc = np.average(data['acc'], weights=[len(year_data) for year_data in test_data.groupby('Season')])
        results[model] = {
            'accuracy': weighted_acc,
            'total_games': data['total']
        }
    
    # Ensemble performance
    results['ensemble'] = {
        'accuracy': accuracy_score(metrics['ensemble']['true'], metrics['ensemble']['pred']),
        'total_games': metrics['total'],
        'correct': metrics['correct']
    }
    
    return results

if __name__ == "__main__":
    # Configuration
    RESULTS_PATH = "data/MNCAATourneyCompactResults.csv"
    TEAMS_PATH = "data/MTeams.csv"
    MODEL_PATH = "tournament_model.pkl"
    
    # Load and preprocess data
    test_data = load_and_preprocess_data(RESULTS_PATH, TEAMS_PATH)
    print(f"Loaded {len(test_data)} historical matchups from {test_data['Season'].min()} to {test_data['Season'].max()}")
    
    # Evaluate model
    results = evaluate_model(MODEL_PATH, test_data)
    
    # Print results
    print("\nModel Performance:")
    for model, metrics in results.items():
        if model == 'ensemble':
            print(f"\nEnsemble Model:")
            print(f"- Accuracy: {metrics['accuracy']:.2%}")
            print(f"- Correct Predictions: {metrics['correct']}/{metrics['total_games']}")
        else:
            print(f"\n{model}:")
            print(f"- Accuracy: {metrics['accuracy']:.2%}")
            print(f"- Total Games Evaluated: {metrics['total_games']}")

    # Calculate theoretical maximum
    print("\nTheoretical Maximum:")
    print(f"- All games have known outcomes (favorite always wins)")
    print(f"- Maximum Possible Accuracy: 100.00%")