import pandas as pd

def load_movies_data(csv_file):
    # Load movies data from CSV file
    movies_df = pd.read_csv(csv_file)
    return movies_df
