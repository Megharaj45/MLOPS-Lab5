from src.features import RatingFeatures

features = RatingFeatures.load("models/rating_features.pkl")

print("Similar users to user 1:")
print(features.get_similar_users(1, 5))