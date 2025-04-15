import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your data
data = pd.read_csv('your_sales_data.csv', parse_dates=['date'])

# Group by SKU (item, store, state) and aggregate sales
grouped_data = data.groupby(['date', 'item', 'store_id', 'state']).sum().reset_index()

# Function to evaluate SARIMA model
def sarima_model(train, test, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    forecast = result.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    return mae, rmse, forecast

# Function to evaluate Prophet model
def prophet_model(train, test):
    train_prophet = train.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train_prophet)
    
    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)
    
    mae = mean_absolute_error(test, forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test, forecast['yhat']))
    
    return mae, rmse, forecast['yhat']

# Function to evaluate XGBoost model
def xgboost_model(train, test):
    for i in range(1, 8):
        train[f'lag_{i}'] = train['sales'].shift(i)
        test[f'lag_{i}'] = test['sales'].shift(i)

    train['rolling_mean_7'] = train['sales'].rolling(window=7).mean()
    test['rolling_mean_7'] = test['sales'].rolling(window=7).mean()
    train['rolling_std_7'] = train['sales'].rolling(window=7).std()
    test['rolling_std_7'] = test['sales'].rolling(window=7).std()
    
    train.dropna(inplace=True)

    X_train = train.drop(columns='sales')
    y_train = train['sales']
    X_test = test.drop(columns='sales')
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    forecast = model.predict(X_test)
    
    mae = mean_absolute_error(test['sales'], forecast)
    rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
    
    return mae, rmse, forecast

# Function to train and test models
def train_and_test_models(train, test):
    sarima_mae, sarima_rmse, sarima_forecast = sarima_model(train, test)
    prophet_mae, prophet_rmse, prophet_forecast = prophet_model(train, test)
    xgb_mae, xgb_rmse, xgb_forecast = xgboost_model(train.reset_index(), test.reset_index())
    
    # Select best model by MAE
    best_model = min([
        ('SARIMA', sarima_mae),
        ('Prophet', prophet_mae),
        ('XGBoost', xgb_mae)
    ], key=lambda x: x[1])[0]
    
    return best_model

# Function to forecast for 2 months
def forecast_next_2_months(best_model, train):
    if best_model == 'SARIMA':
        return sarima_model(train, None, steps=60)[2]
    elif best_model == 'Prophet':
        return prophet_forecast(train, steps=60)
    else:
        return xgboost_forecast(train.reset_index(), steps=60)

# Main logic
forecast_results = []

for (item, store_id, state), group in grouped_data.groupby(['item', 'store_id', 'state']):
    group = group.set_index('date').asfreq('D').fillna(0)
    sales_data = group['sales']
    
    # 1 year train, 1 month blackout, 1 month test
    train_size = 365
    blackout_size = 30
    test_size = 30
    train = sales_data[:train_size]
    test = sales_data[train_size + blackout_size:train_size + blackout_size + test_size]
    
    # Train and test models
    best_model = train_and_test_models(train, test)
    
    # 1 year train, 1 month blackout, 2 month forecast
    train_forecast = sales_data[:train_size]
    forecast = forecast_next_2_months(best_model, train_forecast)
    
    forecast_results.append({
        'item': item,
        'store_id': store_id,
        'state': state,
        'best_model': best_model,
        'forecast': forecast
    })

# Convert to DataFrame
forecast_df = pd.DataFrame(forecast_results)

# Example output
print(forecast_df[['item', 'store_id', 'state', 'best_model', 'forecast']].head())
