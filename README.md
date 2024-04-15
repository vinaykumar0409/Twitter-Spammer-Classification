# **Twitter Spammer Classification**

> Dataset : [twitter_spammer dataset](https://www.kaggle.com/datasets/vinaykumar52/twitter-spammer-classification)

## **Model Deployment**

### Prerequisites
You must have the following packages installed :
1. sklearn
2. pandas
3. numpy
4. TensorFlow
5. matplotlib
6. keras
7. requests
7. Flask

### Deployment Structure
It has three major parts :
1. model.py - This contains code for our Machine Learning model to predict Twitter spammer classification on data.
2. app.py - This contains Flask APIs that receive employee details through GUI or API calls, compute the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow users to enter user details and display the predicted Twitter spammer probability.

Our final Model is in the model.py file

### Running the project

1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using the below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You will be asked to enter the username on the webpage

Enter Twitter username:

* Username: username of a Twitter profile

Now hit the Check Spam button.
If everything goes well, you should  be able to see the output on the HTML page!

> Note: These steps are for deploying a model on localhost.
