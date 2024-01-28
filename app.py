from flask import Flask, render_template, request
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the pre-trained deep learning model

# Load the model using pickle
# with open('pretrained_model.pkl', 'rb') as file:
#     model = pickle.load(file)
model = load_model('full_model.h5')

# Twitter API credentials
rapidapi_key = 'api'
rapidapi_host = 'twitter154.p.rapidapi.com'

def get_user_details(username):
    url = "https://twitter154.p.rapidapi.com/user/details"
    querystring = {"username": username}
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def extract_features(user_details):
    created_at = datetime.strptime(user_details.get('creation_date', ''), "%a %b %d %H:%M:%S +0000 %Y")
    account_age_days = (datetime.utcnow() - created_at).days

    num_followings = user_details.get('following_count', 0)
    num_followers = user_details.get('follower_count', 0)
    num_tweets = user_details.get('number_of_tweets', 0)
    screen_name_length = len(user_details.get('username', ''))
    description_length = len(user_details.get('description', ''))
    followers_to_followings_ratio = num_followers / max(num_followings, 1)
    avg_tweets_per_day = num_tweets / max(account_age_days, 1)

    # You can extract additional features as needed

    features = {
        'NumberOfFollowings': num_followings,
        'NumberOfFollowers': num_followers,
        'NumberOfTweets': num_tweets,
        'LengthOfScreenName': screen_name_length,
        'LengthOfDescriptionInUserProfile': description_length,
        'AccountAge': account_age_days,
        'FollowersToFollowingsRatio': followers_to_followings_ratio,
        'AvgTweetsperDay': avg_tweets_per_day,
        'NumWords' : 14.497286012555257,
        'NumMentions': 0.3618400354716109,
        'NumHashtags': 0.1421296670167871,
        'NumLinks': 0.47065550605149636
    }

    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        username = request.form['username']

        # Get user information from Twitter API
        user_details = get_user_details(username)

        # Extract features
        features = extract_features(user_details)

        # Convert features into a DataFrame
        features_df = pd.DataFrame(features, index = [0])
        first_row = features_df.iloc[0]
        features_df = features_df.append(first_row, ignore_index=True)

        # Make prediction using the deep learning model
        # Assuming features_df has shape (1, num_features), adjust it according to your actual feature dimensions
        features_df_cnn = features_df.values.reshape((features_df.shape[0], features_df.shape[1], 1))
        features_df_rnn = features_df.values.reshape((features_df.shape[0], features_df.shape[1], 1))
        features_df_lstm = features_df.values.reshape((features_df.shape[0], features_df.shape[1], 1))
     

        print(features_df)
        print(features_df_cnn, features_df_rnn, features_df_lstm)


        # Make prediction
        prediction = model.predict([features_df_cnn, features_df_rnn, features_df_lstm])


        print(prediction)
        # Print the model summary to check its architecture
        print(model.summary())

        # Display the prediction and user details on the web page
        return render_template('result.html', username=username, prediction=prediction[0][0], user_details=user_details, features_df=features_df)

if __name__ == '__main__':
    app.run(debug=True)
