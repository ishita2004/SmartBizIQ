from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.cluster import KMeans
import io

router = APIRouter()

@router.post("/segment")
async def segment_customers(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Validate columns
        if df.empty or df.shape[1] < 2:
            return {"error": "CSV must contain at least two columns for clustering."}

        # Apply KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df.select_dtypes(include=['float64', 'int64']))

        # Return sample with cluster labels
        return df.head(10).to_dict(orient="records")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
