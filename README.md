# **Twitter Spammer Classification**

> Dataset : [twitter_spammer dataset](https://www.kaggle.com/datasets/vinaykumar52/twitter-spammer-classification)

## **Model Deployment**

### Prerequisites
You must have following packages installed :
1. sklearn
2. pandas
3. numpy
4. tensorflow
5. matplotlib
6. keras
7. requests
7. Flask

### Deployment Structure
It has three major parts :
1. model.py - This contains code for our Machine Learning model to predict heart failure based on data in 'heart.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter patient details and displays the predicted heart failure probability.

Our final Model is in model.py file

### Running the project

1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You will be asked to enter the username on the webpage

Enter twitter username:

* username : username of a twitter profile

Now hit the Check spam button.
If everything goes well, you should  be able to see the output on the HTML page!

> Note : These steps are for deployment of model on localhost.
