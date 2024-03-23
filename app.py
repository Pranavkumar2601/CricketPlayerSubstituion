from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


knn_model = load('model.pkl')

df = pd.read_csv('cricket_data.csv')

# used this to retrive data from the dataset in html form  input suggestion
player_names = df['Player Name'].unique().tolist()
opposition_teams = df['Opposition Team'].unique().tolist()


def recommend_best_player_knn(player_name, opposition_team):
    
    
    # Find the index of the player in the dataset
    player_index = df[(df['Player Name'] == player_name) & (df['Opposition Team'] == opposition_team)].index
    
    if len(player_index) == 0:
        print("No data found for the specified player name and opposition team combination.")
        return [None]

    player_index = player_index[0]  # Select the first index if multiple rows are found

    # Encode player name and opposition team
    label_encoder_player = LabelEncoder()
    label_encoder_opposition = LabelEncoder()
    df['Player Name Encoded'] = label_encoder_player.fit_transform(df['Player Name'])
    df['Opposition Team Encoded'] = label_encoder_opposition.fit_transform(df['Opposition Team'])

    # Extract features for the player
    player_features = df.iloc[player_index][['Total Boundaries', 'Century', 'Half Century', 'Avg SR', 'Player Name Encoded', 'Opposition Team Encoded']].values.reshape(1, -1)

    # Find the K-nearest neighbors of the player based on their performance against the specified opposition team
    knn_indices = knn_model.kneighbors(player_features, return_distance=False)[0]

    # Get the player names of the nearest neighbors
    nearest_players = df.iloc[knn_indices]['Player Name'].unique()

    return nearest_players.tolist()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html', player_names=player_names, opposition_teams=opposition_teams)

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        player_name = request.form['player_name']
        opposition_team = request.form['opposition_team']
        recommended_best_players = recommend_best_player_knn(player_name, opposition_team)
        return render_template('result.html', player_name=player_name, opposition_team=opposition_team, recommended_players=recommended_best_players)

if __name__ == '__main__':
    app.run(debug=True)