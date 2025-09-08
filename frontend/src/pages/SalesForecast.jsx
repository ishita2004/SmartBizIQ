import React, { useState, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush
} from "recharts";
import { Spinner, Form, Button, Alert, Card, Container, Row, Col, OverlayTrigger, Tooltip as BootstrapTooltip } from "react-bootstrap";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { useDropzone } from "react-dropzone";
import "./SalesForecast.css";

// Custom Tooltip for Recharts with color customization
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const filtered = payload.filter(p => !isNaN(p.value));
    if (!filtered.length) return null;

    return (
      <div
        style={{
          background: "#14191eff",
          border: "1px solid #090f15ff",
          borderRadius: "8px",
          padding: "10px",
          color: "#ffffff",
          boxShadow: "0px 2px 8px rgba(0,0,0,0.2)"
        }}
      >
        <p style={{ margin: 0, fontWeight: "bold" }}>Year: {label}</p>
        {filtered.map((p, idx) => (
          <p key={idx} style={{ color: p.stroke, margin: "4px 0" }}>
            <strong>{p.name}:</strong> ${p.value.toFixed(2)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

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

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    setFile(selectedFile);
    setFileName(selectedFile?.name || "");
    setError("");
    setResult(null);
    setSummary("");
    setHistorical([]);
    setInsights(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: ".csv" });

  const handleUpload = async () => {
    if (!file) {
      alert("📂 Please upload a CSV file before forecasting.");
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

      // Parse uploaded file for historical data
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

      setHistorical(parsed);

      const lastHistoricalYear = Math.max(...parsed.map(p => parseInt(p.ds)));

      // Forecasted values from backend
      const forecasted = res.data.forecast
        .map(item => ({
          ds: typeof item.ds === "string" && item.ds.includes("-")
            ? new Date(item.ds).getFullYear().toString()
            : item.ds.toString(),
          yhat: item.yhat,
          type: "Forecast"
        }))
        .filter(f => parseInt(f.ds) > lastHistoricalYear);

      const fullResult = {
        model,
        forecast: forecasted,
        metrics: res.data.metrics,
        summary: res.data.summary,
        bi_insights: res.data.bi_insights
      };

      setResult(fullResult);

      setAllResults(prev => {
        const others = prev.filter(r => r.model !== model);
        return [...others, fullResult];
      });

      setSummary(res.data.summary);
      setInsights(res.data.bi_insights);
      setError("");
    } catch (err) {
      console.error("Upload failed:", err);
      setError(err.response?.data?.error || "❌ Error processing the CSV file.");
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

  const handleDownloadReport = async () => {
    const container = document.querySelector(".forecast-container");
    if (!container) return;

    try {
      const canvas = await html2canvas(container, { scale: 2 });
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`${model}_forecast_report.pdf`);
    } catch (err) {
      console.error("Report generation failed:", err);
    }
  };

  const forecastedOnly = result?.forecast || [];
  const comparisonYears = Array.from(
    new Set(allResults.flatMap(r => r.forecast).map(f => f.ds))
  );

  return (
    <Container className="forecast-container">
      <Card className="forecast-card">
        <h2 className="forecast-title">📈 Sales Forecasting Dashboard</h2>
        <p className="forecast-subtitle">
          Upload historical sales data and choose a model to predict future years.
          Supported columns: <code>Year, Value</code> or <code>ds, y</code>.
        </p>

        <Form className="mb-4">
          <Form.Group className="mb-3">
            <Form.Label>
              🔧 Choose Forecasting Model{" "}
              <OverlayTrigger placement="right" overlay={<BootstrapTooltip>
                Prophet is best for seasonality. ARIMA is classical. LSTM/GRU use deep learning.
              </BootstrapTooltip>}>
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

          <div {...getRootProps()} className={`upload-box ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="upload-icon">⬆️</div>
            <h5>Upload CSV File</h5>
            <p>Drag & drop or click to browse</p>
            {fileName && <small className="text-muted">Selected File: {fileName}</small>}
          </div>

          <Button variant="dark" className="mt-3" onClick={handleUpload} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : "Upload & Forecast"}
          </Button>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}
        {!result && !loading && <Alert variant="info">
          ℹ️ Upload your CSV and choose a model to start forecasting.
        </Alert>}

        {forecastedOnly.length > 0 && (
          <>
            <h4 className="mt-4 mb-3">📈 Forecast Chart (Future Forecast Only)</h4>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={[...historical, ...forecastedOnly]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ds" />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="yhat"
                  name={`${model.toUpperCase()} Forecast`}
                  stroke="#007bff"
                  dot={false}
                  strokeWidth={2}
                />
                <Brush dataKey="ds" height={30} stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>

            <Row className="mt-4">
              <Col md={6}>
                <Card className="p-3 forecast-data-card">
                  <h5>Forecasted Sales (Next Years)</h5>
                  <ul>
                    {forecastedOnly.map((item, i) => (
                      <li key={i}>
                        <strong>{item.ds}:</strong> ${item.yhat.toFixed(2)}
                      </li>
                    ))}
                  </ul>
                  <p className="text-success">{summary}</p>
                  <Button variant="outline-primary" className="ms-1" onClick={handleDownloadCSV}>⬇️ Download CSV</Button>
                  <Button variant="outline-primary" className="ms-1" onClick={handleDownloadReport}>📄 Download PDF</Button>
                </Card>
              </Col>

              {/* ===== Model Comparison Section ===== */}
              {/* ===== Model Comparison Section ===== */}
<Col md={6}>
  <Card className="p-3 forecast-compare-card" style={{ minHeight: "400px", overflowY: "auto" }}>
    <h5 style={{ color: "#000" }}>Model Comparison</h5>
<table
  className="table table-bordered table-sm model-comparison-table"
  style={{ backgroundColor: "#fff", color: "#000" }} // white background, black text
>
  <thead>
    <tr>
      <th style={{ color: "#000" }}>Year</th>
      {allResults.map(r => (
        <th key={r.model} style={{ color: "#000" }}>{r.model.toUpperCase()}</th>
      ))}
    </tr>
  </thead>
  <tbody>
    {comparisonYears.map((year, i) => (
      <tr key={i}>
        <td style={{ color: "#000" }}>{year}</td>
        {allResults.map(r => {
          const found = r.forecast.find(f => f.ds === year);
          return (
            <td key={r.model} style={{ color: "#000" }} data-label={r.model.toUpperCase()}>
              {found ? `$${found.yhat.toFixed(2)}` : "-"}
            </td>
          );
        })}
      </tr>
    ))}
  </tbody>
</table>


    {/* === Metrics Display === */}
    {allResults.length > 0 && (
      <div className="mt-3">
        <h6 style={{ color: "#fff" }}>Model Metrics (MAE / MSE / RMSE)</h6> {/* keep metrics white */}
        <table className="table table-bordered table-sm" style={{ color: "#fff" }}>
          <thead>
            <tr>
              <th style={{ color: "#fff" }}>Model</th>
              <th style={{ color: "#fff" }}>MAE</th>
              <th style={{ color: "#fff" }}>MSE</th>
              <th style={{ color: "#fff" }}>RMSE</th>
            </tr>
          </thead>
          <tbody>
            {allResults.map(r => (
              <tr key={r.model}>
                <td style={{ color: "#fff" }}>{r.model.toUpperCase()}</td>
                <td style={{ color: "#fff" }}>{r.metrics?.MAE ?? "-"}</td>
                <td style={{ color: "#fff" }}>{r.metrics?.MSE ?? "-"}</td>
                <td style={{ color: "#fff" }}>{r.metrics?.RMSE ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )}
  </Card>
</Col>

            </Row>
          </>
        )}
      </Card>
    </Container>
  );
};

export default SalesForecast;
