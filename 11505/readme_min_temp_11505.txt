* Abu sir has given us raw dataset for minimum temperatures for 4 weather stations of Dhaka city. They are station id : 11111, 41909, 11505, 11513. 
* 1st Nafisa took the raw dataset and preprocessed it. She scaled the dataset (drop.py and Processed_Min_Temp_Data.xlsx) and implemented knn imputation on the raw dataset (knn.py and Processed_Min_Temp_Data_Imputed.xlsx). 
* Then Meherin separate station 11505 data in a file named (min_temp_station_11505.xlsx) . 
* She handled outlier from this file (outlier_min_temp_11505.py and outlier_processed_min_temp_station_11505.xlsx). 
* Then SSA algorithm is implemented in this output file (2d_ssa_min_temp_11505.py and 2d_ssa_reconstructed_11505_min_temp.xlsx). 
* Then Nafisa analyzed some trend, seasonality, noise from this ssa reconstructed data (). 
* In the ssa reconstructed dataset Meherin has implemented Time series split ( K-fold time series split cross validation (EWKCV)----split_min_temp_11505.py and data in "splits_min_temp_11505" folder) for RF and SVR machine learning algorithm. 
* SSA data are normalized (normalize_max_temp_41909.py and data in normalized_max_temp_station_41909.xlsx)
* Then SVR is implemented using this split (svr_max_temp_41909.py and svr_prediction_max_temp_41909 folder for output data). Shows prediction of the test folds data and performance parameters MSE, MAE, RMSE, R^2.
* RF is applied on the dataset (2d_ssa_reconstructed_11505_min_temp) in python file (rf_prediction_11505_min_temp.py) and the output files (test data with prediction) are in (rf_prediction_output_11505_min_temp folder). we've used "k fold time series cross validation" split method for k =7 for best output. The comparison output MAE, MSE, RMSE. R2 are shown in the terminal while running the python file.
