from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample music ratings dataset
data = {
    'user_ids': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'track_ids': ['track1', 'track2', 'track3', 'track1', 'track4', 'track2', 'track5', 'track6', 'track4', 'track7'],
    'ratings': [5, 4, 3, 4, 5, 3, 2, 4, 3, 4]
}

# Create a Pandas DataFrame from the dataset
import pandas as pd
df = pd.DataFrame(data)

# Define the Reader and Dataset from surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_ids', 'track_ids', 'ratings']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Build the collaborative filtering model using KNNBasic
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Make recommendations for a user
def get_track_recommendations(user_id, n=5):
    """
    Get music track recommendations for a given user.
    """
    # Get all track ids
    all_track_ids = df['track_ids'].unique()

    # Remove tracks the user has already rated
    user_rated_tracks = df[df['user_ids'] == user_id]['track_ids'].values
    tracks_to_predict = list(set(all_track_ids) - set(user_rated_tracks))

    # Make predictions for the user on tracks not yet rated
    predicted_ratings = [model.predict(user_id, track_id).est for track_id in tracks_to_predict]

    # Get indices of the top n recommendations
    top_indices = sorted(range(len(predicted_ratings)), key=lambda i: predicted_ratings[i], reverse=True)[:n]

    # Get the track ids of the top recommendations
    top_track_ids = [tracks_to_predict[i] for i in top_indices]

    return top_track_ids

# Example: Get recommendations for user 1
user_id_to_recommend = 1
recommended_tracks = get_track_recommendations(user_id_to_recommend)
print(f"Top 5 track recommendations for user {user_id_to_recommend}: {recommended_tracks}")
