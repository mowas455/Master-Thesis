import numpy as np 
import pandas as pd
import statsmodels.api as sm

#### ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#### Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
   
 # test set arrangements
def to_sequences(dataset, horizon=1):
    x = []

    for i in range(len(dataset)-horizon-1):
        window = dataset[i:(i+horizon)] # make the window of features
        x.append(window)       
    return np.array(x)

#### Root mean squared error or RMSE
def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

#### Mean Absolute Error
def measure_mae(pred,y):
  resid = y - pred
  mae = np.mean(np.abs(resid))
  return mae

#### step mean absolute error
def measure_step_mae(actual,predicted,hz):
  ls = list()
  # iteratre over the columns to find the mean error of forecast
  for i in range(1,hz+1):
    ls.append(measure_mae(actual[i-1],predicted[i-1]))
  return ls


#### step root mean squared error
def measure_step_rmse(actual,predicted,hz):
  ls = list()
  # iteratre over the columns to find the mean error of forecast
  for i in range(1,hz+1):
    ls.append(measure_rmse(actual[i-1],predicted[i-1]))
  return ls
  
#### SARIMAX forecasting using sm.tsa.statespace model
def sarima_forecast(history, config,hz):
    order, sorder, = config
    # define model
    model = sm.tsa.statespace.SARIMAX(history,
                                order=order,
                                seasonal_order=sorder,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # train pred
    train_pred = model_fit.predict()
    # evaluate the train
    train_rmse = measure_rmse(history,train_pred)
    train_mae = measure_mae(history,train_pred)
    # make one step forecast
    ci = model_fit.forecast(hz)
    return ci,train_rmse, train_mae, train_pred

# walk forward validation
def wfv_sarima(train,test,cfg,hz):
  # train and test in array and cfg in list
  prediction = list()
  rmse = list()
  mae = list()
  trmse = list()
  tmae = list()
  tpred = list()
  # seed history with training dataset
  history = train
  # step over each time-step in the test set
  for i in range(1,len(test)+1):
     # fit model and make forecast for history
    yh,tr,tm,tp = sarima_forecast(history,cfg,hz)
    # store forecast in list of predictions
    prediction.append(yh)
    tpred.append(tp)
    # add actual observation to history for the next loop
    history = np.append(history,yh[0])
    # append the train 
    trmse.append(tr)
    tmae.append(tm)
    # evaluate the test set with prediction
    er1 = measure_rmse(yh,test[i-1])
    er2 = measure_mae(yh,test[i-1])
    rmse.append(er1)
    mae.append(er2)
  return prediction,trmse,tmae, rmse, mae,tpred

