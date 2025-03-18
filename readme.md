### To train a new model:

- change dates in tournament_predictor.py
- run:
  ```
  python tournament_predictor.py
  ```
- model is saved to tournament_model.pkl

### To test model on previous years results:

- Change start_year
- run:

```
python model_test.py
```

### To predict matchups for current year:

- edit matchups.csv to match names that are on teamranking.com
  - need to search teamranking.com for appropriate name of each team
- run:

```
python predict_matchup.py matchups.csv predictions.csv --year 2025
```

- change year in above code to match year you want to simulate matchups
- results are in predictions.csv
