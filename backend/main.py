from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import traceback
import io


# ✅ Initialize app
app = FastAPI()

# ✅ Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Label map for customer segmentation
LABEL_MAP = {
    0: "🧊 Low-engagement",
    1: "🎯 VIP",
    2: "🛍️ High-spender",
    3: "🧠 Moderate",
    -1: "🔍 Outlier (DBSCAN)"
}



from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from prophet import Prophet
import pandas as pd
import numpy as np
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
import traceback  # ✅ ADD THIS



@app.post("/forecasting")
async def forecast(
    file: UploadFile = File(...),
    model: str = Query("prophet")  # get model from URL
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        print("🧾 CSV Columns:", df.columns.tolist())
        print("📊 Sample Data:\n", df.head())

        # Normalize columns
        if 'Year' in df.columns and 'Value' in df.columns:
            df['ds'] = pd.to_datetime(df['Year'].astype(str), format='%Y')
            df['y'] = df['Value']
        elif 'ds' in df.columns and 'y' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        else:
            return JSONResponse(status_code=400, content={"error": "CSV must contain either ['Year', 'Value'] or ['ds', 'y'] columns."})

        df = df[['ds', 'y']].dropna()
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "No valid data after parsing the file."})

        # ============================
        # 🔮 Model-Based Forecasting
        # ============================

        if model == "prophet":
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=5, freq='YE')
            forecast = m.predict(future)
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        elif model == "arima":
            df_arima = df.copy().set_index("ds")
            arima = ARIMA(df_arima['y'], order=(1, 1, 1)).fit()
            future_dates = pd.date_range(start=df['ds'].max() + pd.DateOffset(years=1), periods=5, freq='YE')
            forecast_values = arima.forecast(steps=5)
            result = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_values,
                'yhat_lower': forecast_values,
                'yhat_upper': forecast_values
            })
        elif model == "lstm":
            raise NotImplementedError("LSTM model is not implemented yet.")
        elif model == "gru":
            raise NotImplementedError("GRU model is not implemented yet.")
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported model: {model}"})

        # ======================================
        # 📊 Business Insights & Evaluation
        # ======================================
        result['cumulative_yhat'] = result['yhat'].cumsum()
        forecast_data = result.tail(5)
        forecast_data['ds'] = forecast_data['ds'].dt.year

        # Merge for metrics
        try:
            merged = pd.merge(df, result[['ds', 'yhat']], on='ds', how='inner')
            mae = mean_absolute_error(merged['y'], merged['yhat'])
            mse = mean_squared_error(merged['y'], merged['yhat'])
            rmse = np.sqrt(mse)
        except:
            mae = mse = rmse = 0

        # Best/Worst Year
        historical = df.copy()
        historical['year'] = historical['ds'].dt.year
        best_year = int(historical.loc[historical['y'].idxmax(), 'year'])
        worst_year = int(historical.loc[historical['y'].idxmin(), 'year'])

        mean_val = historical['y'].mean()
        std_val = historical['y'].std()
        historical['z_score'] = (historical['y'] - mean_val) / std_val
        outliers = historical[np.abs(historical['z_score']) > 2]['year'].astype(int).tolist()

        # Summary
        growth_pct = ((forecast_data['yhat'].iloc[-1] - forecast_data['yhat'].iloc[0]) / forecast_data['yhat'].iloc[0]) * 100
        summary = f"Sales are projected to reach ${forecast_data['yhat'].iloc[-1]:.2f} by {forecast_data['ds'].iloc[-1]}, growing {growth_pct:.1f}% over 5 years."

        return {
            "forecast": forecast_data.to_dict(orient='records'),
            "metrics": {
                "MAE": round(mae, 2),
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 2)
            },
            "summary": summary,
            "bi_insights": {
                "best_year": best_year,
                "worst_year": worst_year,
                "outliers": outliers
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )



# ✅ Customer Segmentation Endpoint with Segment Summaries
@app.post("/segmentation/segment-customers")
async def segment_customers(file: UploadFile = File(...), method: str = Query("kmeans")):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if not {'Age', 'Annual_Income', 'Spending_Score'}.issubset(df.columns):
            return JSONResponse(status_code=400, content={"error": "CSV must have Age, Annual_Income, Spending_Score"})

        features = df[['Age', 'Annual_Income', 'Spending_Score']]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        # Clustering
        if method.lower() == "dbscan":
            model = DBSCAN(eps=1.2, min_samples=2)
        else:
            model = KMeans(n_clusters=3, random_state=42)

        clusters = model.fit_predict(scaled)
        df['Cluster'] = clusters
        df['Label'] = df['Cluster'].map(LABEL_MAP).fillna("🧠 Moderate")

        # 📊 Cluster Visualization
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='Annual_Income', y='Spending_Score', hue='Label', data=df,
            palette='Set2', s=100, alpha=0.8
        )
        plt.title("Customer Segmentation Clusters")
        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 📢 Segment Summaries
        summaries = {}
        for label, group in df.groupby("Label"):
            count = len(group)
            age_range = f"{group['Age'].min()}–{group['Age'].max()}"
            income_range = f"₹{group['Annual_Income'].min():,} to ₹{group['Annual_Income'].max():,}"
            score_range = f"{group['Spending_Score'].min()} to {group['Spending_Score'].max()}"
            summaries[label] = f"{label} ({count} customers): Typically aged {age_range}, incomes from {income_range}, and spending scores between {score_range}."

        return JSONResponse(content={
            "data": df[['Age', 'Annual_Income', 'Spending_Score', 'Cluster', 'Label']].to_dict(orient='records'),
            "plot": img_base64,
            "summaries": summaries
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/churn-prediction")
async def churn_prediction(file: UploadFile = File(...), model: str = Query("random_forest")):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if 'Churn' not in df.columns:
            return JSONResponse(status_code=400, content={"error": "CSV must contain a 'Churn' column."})

        # ✅ Convert 'Yes'/'No' to 1/0 for classification
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # ✅ Ensure conversion succeeded
        if df['Churn'].isnull().any():
            return JSONResponse(status_code=400, content={"error": "Churn column has invalid values. Use only 'Yes' or 'No'."})

        # Split into features and target
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        # ✅ One-hot encode categorical variables
        X = pd.get_dummies(X)

        # Split train-test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model
        if model == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()

        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred).tolist()

        return JSONResponse(content={
            "classification_report": report,
            "confusion_matrix": matrix
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/anomaly-detection")
async def detect_anomalies(file: UploadFile = File(...), method: str = Query("isolation_forest")):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Ensure numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return JSONResponse(status_code=400, content={"error": "CSV must contain numeric features."})

        # Use only numeric data
        X = numeric_df.values

        # Choose algorithm
        if method == "svm":
            model = OneClassSVM(gamma='auto')
        elif method == "zscore":
            z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
            anomalies = (z_scores > 3).any(axis=1)
        else:  # Default: Isolation Forest
            model = IsolationForest(contamination=0.1, random_state=42)
            anomalies = model.fit_predict(X) == -1

        df['Anomaly'] = anomalies.astype(int)

        # 📊 Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df.index, numeric_df.iloc[:, 0], c=df['Anomaly'], cmap='coolwarm', s=50)
        plt.title("Anomaly Detection")
        plt.xlabel("Index")
        plt.ylabel(numeric_df.columns[0])
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return JSONResponse(content={
            "data": df.to_dict(orient='records'),
            "plot": img_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@app.post("/recommendation")
async def recommend_products(file: UploadFile = File(...), customer_id: int = Query(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if not {'CustomerID', 'Product', 'Category', 'Rating'}.issubset(df.columns):
            return JSONResponse(status_code=400, content={"error": "CSV must contain CustomerID, Product, Category, Rating columns."})

        # Create a product profile by combining category and product name
        df['product_profile'] = df['Product'] + " " + df['Category']

        # Pivot table for user-item matrix
        user_product_matrix = df.pivot_table(index='CustomerID', columns='product_profile', values='Rating', fill_value=0)

        # Compute similarity matrix
        similarity = cosine_similarity(user_product_matrix)

        # Get similar customers
        customer_idx = user_product_matrix.index.get_loc(customer_id)
        sim_scores = list(enumerate(similarity[customer_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # exclude self

        # Recommend products liked by similar users
        similar_customers = [user_product_matrix.index[i] for i, _ in sim_scores[:2]]
        recommended_products = df[df['CustomerID'].isin(similar_customers)]

        # Filter out already rated products
        rated_products = df[df['CustomerID'] == customer_id]['Product'].tolist()
        recommended_products = recommended_products[~recommended_products['Product'].isin(rated_products)]

        return JSONResponse(content={
            "customer_id": customer_id,
            "recommendations": recommended_products[['Product', 'Category']].drop_duplicates().to_dict(orient='records')
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
