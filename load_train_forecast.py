import sidfc_functions as f

file_path = "sidfc-data/train.csv"
cutoff_date = '2016-12-31'
forecast_save_path = 'sidfc-app/Data/'

[X_train, 
 y_train, 
 Item_Store_Coordinates, 
 forecast_index, 
 sc] = f.load_data(file_path, 
                   cutoff_date, 
                   in_seq_length = 365, 
                   out_seq_length = 365)

PRED_CURVES = f.model_preds(X_train, 
                            y_train, 
                            forecast_index, 
                            sc,
                            layer_size = 256,
                            iterations = 30,
                            epochs = 30,
                            batch_size = 64)

forecast = f.get_forecast(PRED_CURVES,
                          Item_Store_Coordinates,
                          forecast_index,
                          forecast_save_path)