import pandas as pd

# Fájlok beolvasása
predictions_df = pd.read_csv('predictions.csv')
example_solutions_df = pd.read_csv('example_solutions.csv')

# Rendezés az example_solutions.csv Id oszlopának sorrendje szerint
sorted_predictions_df = predictions_df.set_index('Id').reindex(example_solutions_df['Id']).reset_index()

# Mentés az új fájlba
sorted_predictions_df.to_csv('sorted_predictions.csv', index=False)

print("A predictions.csv sikeresen rendezve lett az example_solutions.csv sorrendje alapján.")
