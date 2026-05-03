import pandas as pd
from src.features import RatingFeatures

def main():
    print("Loading data...")
    df = pd.read_csv("data/processed/ratings_clean.csv")

    print("Building features...")
    features = RatingFeatures()
    features.fit(df)

    print("Saving features...")
    features.save("models/rating_features.pkl")

    print("Done!")

if __name__ == "__main__":
    main()