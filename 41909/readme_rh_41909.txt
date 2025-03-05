* Abu sir has given us raw dataset for relative humidity for 4 weather stations of Dhaka city. They are station id : 11111, 41909, 11505, 11513. 
* 1st Nafisa took the raw dataset and preprocessed it. She scaled the dataset and implemented knn imputation on the raw dataset (knn.py and Processed_Humidity_Data_Stationwise.xlsx). 
* Then Meherin separate station 41909 data in a file named (rh_station_41909.xlsx) . 
* She handled outlier from this file (outlier_rh_41909.py and outlier_processed_rh_station_41909.xlsx). 
* Then SSA algorithm is implemented in this output file (2d_ssa_rh_41909.py and 2d_ssa_reconstructed_41909_rh.xlsx). 
* Then Nafisa analyzed some trend, seasonality, noise from this ssa reconstructed data (). 
* In the ssa reconstructed dataset Meherin has implemented Time series split ( K-fold time series split cross validation (EWKCV)----split_rh_41909.py and data in "splits_max_rh_41909" folder) for RF and SVR machine learning algorithm. 
* SSA data are normalized (normalize_max_temp_41909.py and data in normalized_max_temp_station_41909.xlsx)
* Then SVR is implemented using this split (svr_max_temp_41909.py and svr_prediction_max_temp_41909 folder for output data). Shows prediction of the test folds data and performance parameters MSE, MAE, RMSE, R^2.
* RF is applied on the dataset (2d_ssa_reconstructed_41909) in python file (rf_prediction_41909_rh.py) and the output files (test data with prediction) are in (rf_prediction_output_41909_rh folder). We've used "k fold time series cross validation" split method for k =7 for best output. The comparison output MAE, MSE, RMSE. R2 are shown in the terminal while running the python file.
