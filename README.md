# sidfc
About: This is my solution to Kaggle’s Store Item Demand Forecasting Challenge 
 
https://www.kaggle.com/c/demand-forecasting-kernels-only/overview
 
An ensemble of LSTMs is trained and used to forecast product demand with varying confidence. The results are shown on a dashboard style website. 
 
This repo is subdivided in two parts: 
 
ML/DL:
 
1 – download “Train.csv” from - https://www.kaggle.com/c/demand-forecasting-kernels-only/data and include it in the folder “sidfc-data”
 
2 – On the command line, run “load_train_forecast.py”. 
 
requirements.txt contains all that is needed to in order to run the above.
 
This will load the data, prepare it, train the models, perform the predictions and save them as a forecast for app digestion.
 
Dash app: 

The app self-contained in “sidfc-app”. To deploy locally, simply navigate to this folder and run “app.py”
 
Alternatively, a Docker container can be pulled from,
 
https://hub.docker.com/repository/docker/agfernandes/sidfc-app
 
Which can be run locally on any Linux supporting machine.
