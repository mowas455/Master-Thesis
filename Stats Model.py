import numpy as np
import pandas as pd
import tensorflow as tf

#### Evaluate the Performance

def evalutate_performance(y_pred, y, name=None):
    """This function evaluates and prints the MSE and MAE of the prediction.
    
    Parameters
    ----------
    y_pred : ndarrary
        Array of size (n,) with predictions.
    y : ndarray
        Array of size (n,) with target values.
    split_time : int
        The leading number of elements in y_pred and y that belong to the training data set.
        The remaining elements, i.e. y_pred[split_time:] and y[split_time:] are treated as test data.
    """
  
    # Compute error in prediction
    resid = y - y_pred
    
    # We evaluate the MSE and MAE in the original scale of the data, i.e. we add back MAX_VAL
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
   
    
    # Print
    print(f'Model {name}\n  Test RMSE: {rmse:.4f},   MAE: {mae:.4f}\n  ')


#### supervised data preparation
    
def univariate_data_prep_function(dataset, start, end, window, horizon):
    '''
    Prepare univariate data that is suitable for a time series
    
    Args:
        dataset (float64): Scaled values for the dependent variable, numpy array of floats 
        start (int): Start point of range, integer
        end (int): End point of range, integer
        window (int): Number of units to be viewed per step, integer
        horizon (int): Number of units to be predicted, integer
    
    Returns:
        X (float64): Generated X-values for each step, numpy array of floats
        y (float64): Generated y-values for each step, numpy array of floats
    '''   
    X = []
    y = []

    start = start + window
    if end is None:
        end = len(dataset) - horizon

    for i in range(start, end):
        indicesx = range(i-window, i)
        X.append(np.reshape(dataset[indicesx], (window, 1)))
        indicesy = range(i,i+horizon)
        y.append(dataset[indicesy])
    return np.array(X), np.array(y)

#### LSTM MODEL

def sss_model(x_scaled,univar_hist_window_sss,horizon_sss,train_split_sss,n_epoch):
    BATCH_SIZE_sss = 32
    BUFFER_SIZE_sss = 15
    x_train_uni_sss, y_train_uni_sss = univariate_data_prep_function(x_scaled, 0, train_split_sss, 
                                                                 univar_hist_window_sss, horizon_sss)

    x_val_uni_sss, y_val_uni_sss = univariate_data_prep_function(x_scaled, train_split_sss, None, 
                                                             univar_hist_window_sss, horizon_sss)
    print ('Length of first Single Window:')
    print (len(x_train_uni_sss[0]))
    print()
    print ('Target horizon:')
    print (y_train_uni_sss[0])

    # tensor slices of train data
    train_univariate_sss = tf.data.Dataset.from_tensor_slices((x_train_uni_sss, y_train_uni_sss))
    train_univariate_sss = train_univariate_sss.cache().shuffle(BUFFER_SIZE_sss).batch(BATCH_SIZE_sss).repeat()
    # tensor slices of validation data
    validation_univariate_sss = tf.data.Dataset.from_tensor_slices((x_val_uni_sss, y_val_uni_sss))
    validation_univariate_sss = validation_univariate_sss.batch(BATCH_SIZE_sss).repeat()
  
    # model fitting parameters
    n_steps_per_epoch = 100
    n_validation_steps = 5
    n_epochs = n_epoch
  
  # tensorflow lstm model
    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(200, input_shape=x_train_uni_sss.shape[-2:],return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=50,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=horizon_sss)])
    model.compile(loss='mse',
              optimizer='adam')
  
  # fit the  model
    history = model.fit(train_univariate_sss, epochs=n_epochs, steps_per_epoch=n_steps_per_epoch,
                    validation_data=validation_univariate_sss, validation_steps=n_validation_steps, verbose =1)
    return model, history,x_train_uni_sss,y_train_uni_sss,x_val_uni_sss,y_val_uni_sss

#### Walk Forward Validation for LSTM model

def walk_forward_valid(train_data, scaler_x, model_s, nsteps, univar_hist_window_sss):
    df_close = train_data
    test_horizon = df_close.tail(univar_hist_window_sss)
    test_history = test_horizon.values
    result = []
    # Define Forecast length here
    window_len = nsteps
    test_scaled = scaler_x.fit_transform(test_history.reshape(-1, 1))

    for i in range(1, window_len+1):
        test_scaled = test_scaled.reshape((1, test_scaled.shape[0], 1))
    
        # Inserting the model
        predicted_results = model_s.predict(test_scaled)
   
        #print(f'predicted : {predicted_results}')
        result.append(predicted_results[0])
        test_scaled = np.append(test_scaled[:,1:],[[predicted_results]])
    result_inv_trans = scaler_x.inverse_transform(result)
    return result_inv_trans


