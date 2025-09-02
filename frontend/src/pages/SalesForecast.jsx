import React, { useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import { Spinner, Form, Button, Alert, Card, Container, OverlayTrigger, Tooltip as BootstrapTooltip } from "react-bootstrap";

const SalesForecast = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [model, setModel] = useState("prophet");
  const [result, setResult] = useState(null);
  const [historical, setHistorical] = useState([]);
  const [summary, setSummary] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [allResults, setAllResults] = useState([]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setFileName(selectedFile?.name || "");
    setError("");
    setResult(null);
    setSummary("");
    setHistorical([]);
    setInsights(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("📂 Please select a CSV file before uploading.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post(
        `http://localhost:8000/forecasting?model=${model}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

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

      const fullResult = {
        model,
        forecast: forecasted,
        metrics: res.data.metrics,
        summary: res.data.summary,
        bi_insights: res.data.bi_insights
      };

      setHistorical(parsed);
      setResult(fullResult);
      setAllResults(prev => [...prev, fullResult]);
      setSummary(res.data.summary);
      setInsights(res.data.bi_insights);
      setError("");
    } catch (err) {
      console.error("Upload failed:", err);
      setError(err.response?.data?.error || "❌ Something went wrong. Please check your CSV format and try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    if (!result?.forecast?.length) return;
    const csvContent = "Year,Forecasted Sales\n" +
      result.forecast.map(row => `${row.ds},${row.yhat.toFixed(2)}`).join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${model}_forecast.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const combinedData = [...historical, ...(result?.forecast || [])];

  return (
    <Container className="my-5">
      <Card className="shadow-lg border-0 p-4">
        <h2 className="mb-4" style={{ color: "#000" }}>📊 Sales Forecasting Dashboard</h2>
        <p className="text-muted">Upload historical sales data and choose a model to predict the next 5 years. Supported columns: <code>Year, Value</code> or <code>ds, y</code>.</p>

        <Form className="mb-4">
          <Form.Group className="mb-3">
            <Form.Label className="text-dark">
              🔧 Choose Forecasting Model{" "}
              <OverlayTrigger placement="right" overlay={<BootstrapTooltip>Prophet is best for business seasonality. ARIMA is classic. LSTM/GRU use deep learning.</BootstrapTooltip>}>
                <span style={{ cursor: "help", color: "#0d6efd" }}>ⓘ</span>
              </OverlayTrigger>
            </Form.Label>
            <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="prophet">Prophet</option>
              <option value="arima">ARIMA</option>
              <option value="lstm">LSTM (Deep Learning)</option>
              <option value="gru">GRU (Deep Learning)</option>
            </Form.Select>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label className="text-dark">📂 Upload CSV File</Form.Label>
            <Form.Control type="file" accept=".csv" onChange={handleFileChange} />
            {fileName && <small className="text-muted">Selected File: {fileName}</small>}
          </Form.Group>

          <Button variant="dark" onClick={handleUpload} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : "Upload & Forecast"}
          </Button>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}
        {!result && !loading && <Alert variant="info">ℹ️ Upload your CSV and choose a model to start forecasting.</Alert>}

        {result?.forecast && (
          <>
            <h4 className="mb-3 text-dark">📈 Forecast Chart (Actual vs Predicted)</h4>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={combinedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ds" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="yhat" stroke="#007bff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>

            <Card className="mt-4 p-3 bg-light">
              <h5 className="text-dark">📅 Forecasted Sales (Next 5 Years)</h5>
              <ul className="text-dark">
                {result.forecast.map((item, i) => (
                  <li key={i}><strong>{item.ds}:</strong> ${item.yhat.toFixed(2)}</li>
                ))}
              </ul>
              <p className="text-success">{summary}</p>
              <Button variant="outline-dark" onClick={handleDownloadCSV}>
                ⬇️ Download Forecast CSV
              </Button>
            </Card>

            <Card className="mt-4 p-3">
              <h5 className="text-dark">📐 Accuracy Metrics</h5>
              <ul className="text-dark">
                <li><strong>MAE:</strong> Mean Absolute Error – average error size: {result.metrics?.MAE?.toFixed(2)}</li>
                <li><strong>MSE:</strong> Mean Squared Error – error squared: {result.metrics?.MSE?.toFixed(2)}</li>
                <li><strong>RMSE:</strong> Root Mean Squared Error – error spread: {result.metrics?.RMSE?.toFixed(2)}</li>
              </ul>
            </Card>

            {insights && (
              <Card className="mt-4 p-3 bg-light border-info">
                <h5 className="text-dark">📌 Business Insights</h5>
                <p className="text-dark"><strong>📈 Best Performing Year:</strong> {insights.best_year}</p>
                <p className="text-dark"><strong>📉 Lowest Sales Year:</strong> {insights.worst_year}</p>
                <p className="text-dark"><strong>🚨 Outlier Years:</strong> {insights.outliers.length ? insights.outliers.join(", ") : "None Detected"}</p>
              </Card>
            )}

            {allResults.length > 1 && (
              <Card className="mt-5 p-3">
                <h4 className="text-dark">🧠 Compare Models</h4>
                <table className="table table-striped mt-3">
                  <thead className="table-dark">
                    <tr>
                      <th>Model</th>
                      <th>MAE</th>
                      <th>MSE</th>
                      <th>RMSE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allResults.map((r, i) => (
                      <tr key={i}>
                        <td className="text-dark">{r.model.toUpperCase()}</td>
                        <td className="text-dark">{r.metrics.MAE?.toFixed(2)}</td>
                        <td className="text-dark">{r.metrics.MSE?.toFixed(2)}</td>
                        <td className="text-dark">{r.metrics.RMSE?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}
          </>
        )}
      </Card>
    </Container>
  );
};

export default SalesForecast;
