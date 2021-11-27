# Demand Forecasting with LSTM ensemble

In this project I use Deep Learning to forecast sales of 50 individual items across 10 stores over a year found in:

 <https://www.kaggle.com/c/demand-forecasting-kernels-only/overview>

I tackle two main problems common in time series analysis: **Dealing with uncertainty** and **effective visualization of large datasets** by modeling data noise with model variance and displaying results utilizing a dash app.

## Technogies
To create this solution, I used
* Python
* Pandas
* Tensorflow (GPU)
* Plotly
* Flask
* Dash
* Docker
* Azure App Services

Additional detailed requirement files are contained in this repository (see below)

## Repository Structure

Organized in two substructures
#### Deep Learning Solution
* /load_train_forecast.py - Main executable. Loads historical sales data, parses it, trains ensemble, computes and pushes resulting forecast for app digestion

#### Dash App
* /sidfc-app/ - This directory is the self contained source code for dash.

  A copy of the app can be downloaded as a docker container at:

  <https://hub.docker.com/repository/docker/agfernandes/sidfc-app>

  Additionally, a running instance of the dashboard was deployed as an azure app here:

  <https://sidfc-dashboard.azurewebsites.net/>

  Which can be run locally on any Linux supporting machine.
