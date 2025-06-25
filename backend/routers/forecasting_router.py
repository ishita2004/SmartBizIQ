from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import io

router = APIRouter()

@router.post("/")
async def forecast(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        print("🧾 CSV Columns:", df.columns.tolist())
        print("📊 Sample Data:\n", df.head())

        # Accept both "Year,Value" or "ds,y"
        if 'Year' in df.columns and 'Value' in df.columns:
            df['ds'] = pd.to_datetime(df['Year'].astype(str), format='%Y')
            df['y'] = df['Value']
        elif 'ds' in df.columns and 'y' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            return JSONResponse(status_code=400, content={
                "error": "CSV must contain columns: either ['Year', 'Value'] or ['ds', 'y']"
            })

        df = df[['ds', 'y']].dropna()

        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid data rows found after parsing."})

        # Fit Prophet
        model = Prophet()
        model.fit(df)

        # Forecast next 5 years (using year end frequency)
        future = model.make_future_dataframe(periods=5, freq='YE')
        forecast = model.predict(future)

        # Accuracy Metrics (on known data)
        merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        mse = mean_squared_error(merged['y'], merged['yhat'])
        rmse = np.sqrt(mse)

        # Extract only future forecasts
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
        forecast_data['ds'] = forecast_data['ds'].dt.year  # Show only year

        # Summary
        last_year = forecast_data['ds'].iloc[-1]
        last_forecast = forecast_data['yhat'].iloc[-1]
        summary = f"Sales expected to reach ${last_forecast:.2f} by {last_year}."

        return {
            "forecast": forecast_data.to_dict(orient='records'),
            "metrics": {
                "MAE": round(mae, 2),
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 2)
            },
            "summary": summary
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
