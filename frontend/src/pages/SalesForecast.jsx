import React, { useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const SalesForecast = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [historical, setHistorical] = useState([]);
  const [summary, setSummary] = useState("");
  const [model, setModel] = useState("prophet");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        `http://localhost:8000/forecasting?model=${model}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      // Parse uploaded file as historical data
      const fileText = await file.text();
      const rows = fileText.trim().split("\n");
      const headers = rows[0].split(",").map(h => h.trim().toLowerCase());
      const dataRows = rows.slice(1);

      const parsed = dataRows.map(row => {
        const values = row.split(",");
        const year = values[headers.indexOf("year")] || values[headers.indexOf("ds")];
        const val = values[headers.indexOf("value")] || values[headers.indexOf("y")];
        return {
          ds: year.trim(),
          yhat: parseFloat(val),
          type: "Historical"
        };
      });

      const forecasted = res.data.forecast.map(item => ({
        ds: typeof item.ds === "string" && item.ds.includes("-")
          ? new Date(item.ds).getFullYear().toString()
          : item.ds.toString(),
        yhat: item.yhat,
        type: "Forecast"
      }));

      setHistorical(parsed);
      setResult({
        forecast: forecasted,
        metrics: res.data.metrics
      });
      setSummary(res.data.summary);
    } catch (err) {
      console.error("Upload failed:", err);
      alert(err.response?.data?.error || "Something went wrong. Please check your CSV and try again.");
    }
  };

  const combinedData = [...historical, ...(result?.forecast || [])];

  return (
    <div style={{ padding: "20px", maxWidth: "900px", margin: "0 auto", fontFamily: "Arial, sans-serif" }}>
      <h2>📊 Sales Forecasting</h2>

      <div className="mb-3">
        <label>Select Forecasting Model:</label>
        <select className="form-control" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="prophet">Prophet</option>
          <option value="arima">ARIMA</option>
          <option value="lstm">LSTM</option>
          <option value="gru">GRU</option>
        </select>
      </div>

      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="form-control mb-3"
      />

      <button className="btn btn-primary" onClick={handleUpload}>
        Upload & Forecast
      </button>

      {result?.forecast && (
        <>
          <h4 className="mt-4">📈 Forecast Visualization</h4>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={combinedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ds" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="yhat" stroke="#007bff" name="Sales" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>

          <div className="mt-4 bg-light p-3 border rounded">
            <h5>🔮 Forecasted Sales (Next 5 Years)</h5>
            <ul style={{ listStyle: "none", paddingLeft: 0 }}>
              {result.forecast.map((item, index) => (
                <li key={index} style={{ padding: "6px 0", borderBottom: "1px dashed #ccc" }}>
                  <strong>{item.ds}:</strong> ${item.yhat.toFixed(2)}
                </li>
              ))}
            </ul>
            {summary && (
              <p className="mt-2 text-success">{summary}</p>
            )}
          </div>

          <div className="mt-3 bg-warning p-3 border rounded">
            <h5>📐 Forecast Accuracy Metrics</h5>
            <ul style={{ listStyle: "none", paddingLeft: 0 }}>
              <li><strong>MAE:</strong> {result.metrics?.MAE?.toFixed(2)}</li>
              <li><strong>MSE:</strong> {result.metrics?.MSE?.toFixed(2)}</li>
              <li><strong>RMSE:</strong> {result.metrics?.RMSE?.toFixed(2)}</li>
            </ul>
            <small className="text-muted">Lower values mean better forecasting accuracy.</small>
          </div>
        </>
      )}
    </div>
  );
};

export default SalesForecast;
