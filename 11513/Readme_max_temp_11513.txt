* Abu sir has given us raw dataset for maximum temperatures for 4 weather stations of Dhaka city. They are station id : 11111, 41909, 11505, 11513. 
* 1st Nafisa took the raw dataset and preprocessed it. She scaled the dataset (drop.py and Processed_Max_Temp_Data.xlsx) and implemented knn imputation on the raw dataset (knn.py and Processed_Max_Temp_Data_Imputed.xlsx). 
* Then Meherin separate station 11505 data in a file named (max_temp_station_11505.xlsx) . 
* She handled outlier from this file (outlier_11513.py and outlier_processed_max_temp_station_11513.xlsx). 
* Then SSA algorithm is implemented in this output file (2d_ssa_11513.py and 2d_ssa_reconstructed_11513_max_temp.xlsx). 
* Then Nafisa analyzed some trend, seasonality, noise from this ssa reconstructed data (). 
* In the ssa reconstructed dataset Meherin has implemented Time series split ( K-fold time series split cross validation (EWKCV)----split_max_temp_11505.py and data in "splits_max_temp_11505+" folder) for RF and SVR machine learning algorithm. 
* SSA data are normalized (normalize_max_temp_11513.py and data in normalized_max_temp_station_11513.xlsx)
* Then SVR is implemented using this split (svr_max_temp_11513.py and svr_prediction_max_temp_11513 folder for output data). Shows prediction of the test folds data and performance parameters MSE, MAE, RMSE, R^2.
