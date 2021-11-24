import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from tensorflow.keras import initializers
import plotly.express as px
import plotly.colors as c
import plotly.graph_objects as go
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, TimeDistributed
import numpy as np
from numpy import array
import keras.backend as K

import warnings
warnings.filterwarnings('ignore')

def load_data(file_path, 
			  cutoff_date, 
			  in_seq_length = 365, 
			  out_seq_length = 365,):

	df = pd.read_csv(file_path)
	df.index = pd.to_datetime(df['date'])
	train_data, test_data = df.loc[:cutoff_date], df.loc[df.index > pd.to_datetime(cutoff_date)]

	N_Items = train_data.item.max()
	N_Stores = train_data.store.max()
	features = N_Items*N_Stores

	nT = len(train_data['sales'].loc[(train_data.item == N_Items) & (train_data.store == N_Stores)])
	train_series_arr = np.zeros([nT,features])

	cc = 0
	Item_Store_Coordinates = []
	for item in range(1,N_Items + 1):
	    for store in range(1,N_Stores + 1):
	        train_series_arr[:,cc] = train_data['sales'].loc[(train_data.item == item) & (train_data.store == store)]
	        Item_Store_Coordinates.append([cc,item,store])
	        cc += 1
	        
	train_series_arr.shape
	test_index = test_data.loc[(test_data.item == item) & (test_data.store == store)].index
	Item_Store_Coordinates = pd.DataFrame(Item_Store_Coordinates)
	Item_Store_Coordinates.columns = ['idx','item','store']
	forecast_index = pd.date_range(start=cutoff_date, periods=out_seq_length + 1, freq='D')[1:]


	sc = MinMaxScaler(feature_range=(-1,1))
	training_set_scaled = sc.fit_transform(train_series_arr)

	seq_length = in_seq_length
	out_seq_length = out_seq_length


	X_train = []
	y_train = []
	for i in range(seq_length, len(training_set_scaled) - out_seq_length):
	    X_train.append(training_set_scaled[i-seq_length:i,:])
	    y_train.append(training_set_scaled[i : i+out_seq_length,:])
	X_train, y_train = np.array(X_train), np.array(y_train)

	print("----------------DATA LOADED FROM '" + file_path + "' WITH TRAINING CUTOFF DATE '" + cutoff_date + "'----------------")

	return X_train, y_train, Item_Store_Coordinates, forecast_index, sc

	

def model_preds(X_train, 
		        y_train, 
			    forecast_index, 
			    sc,
			    layer_size = 256,
			    iterations = 30,
			    epochs = 30,
			    batch_size = 64):

	initializer = tf.keras.initializers.RandomNormal(mean=0., stddev= 1.0)
	LAYER_SIZE = layer_size
	ITERATIONS = iterations

	PRED_CURVES = []
	for cc in range(0,ITERATIONS):
	    
	    print("----------------TRAINING MODEL VARIANT #" + str(cc + 1) + "/" + str(ITERATIONS) + "----------------")

	    input_data = Input(shape=(X_train.shape[1],X_train.shape[2]))
	    lstm1 = LSTM(LAYER_SIZE,
	                 return_sequences = True)
	    lstm2 = LSTM(LAYER_SIZE,
	                 return_sequences = True)
	    dense = TimeDistributed(Dense(X_train.shape[2], 
	                                  activation='linear',
	                                  kernel_initializer= initializer))

	    lstm1_outputs = lstm1(input_data)
	    lstm2_outputs = lstm2(lstm1_outputs)
	    dense_out = dense(lstm2_outputs)

	    regressor = Model(inputs=[input_data],outputs=[dense_out])
	    regressor.compile(optimizer='RMSprop', 
	                      loss='mean_squared_error')

	    regressor.fit(X_train,
	                  y_train, 
	                  epochs=epochs, 
	                  batch_size=batch_size, 
	                  validation_split=0.1)
	    
	    pred_curves = []
	    pred = regressor.predict(X_train[-1].reshape((1, X_train.shape[1], X_train.shape[2])))
	    
	    for curve in range(0,X_train.shape[2]):
	        pred_curve = pd.DataFrame(sc.inverse_transform(pred[0,:,:])[:,curve].ravel())
	        pred_curve.index = forecast_index
	        pred_curves.append(pred_curve)
	    PRED_CURVES.append(pred_curves)

	return PRED_CURVES


def get_forecast(PRED_CURVES,
				 Item_Store_Coordinates,
				 forecast_index,
				 save_path):


	forecast = pd.DataFrame([])

	for store in range(1,Item_Store_Coordinates.store.max() +1):
	    for item in range(1,Item_Store_Coordinates.item.max() +1):

	        N = list(Item_Store_Coordinates.idx.loc[(Item_Store_Coordinates.store == store) &
	                                                (Item_Store_Coordinates.item == item)])[0]
	        ensemble = []
	        for pred in PRED_CURVES:
	            ensemble.append(list(pred[N][0]))
	        ensemble = pd.DataFrame(ensemble).transpose()
	        ensemble.index = pred[0].index

	        temp = pd.DataFrame([])
	        temp['mean_forecast'] = ensemble.mean(axis = 1)
	        temp['std'] = ensemble.std(axis = 1)
	        temp['store'] = store
	        temp['item'] = item

	        forecast = forecast.append(temp)


	forecast_file = 'forecast_' + str(forecast_index[0].month) + "_" + str(forecast_index[0].year) +'.pkl'
	forecast.to_pickle(save_path + forecast_file)

	print("----------------FORECAST SAVED TO '" + save_path + forecast_file + "' ----------------")

	return forecast