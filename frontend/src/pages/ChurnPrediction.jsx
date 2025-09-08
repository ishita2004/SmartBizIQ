import React, { useState, useCallback } from "react";
import axios from "axios";
import { Container, Card, Form, Button, Table, Alert, Spinner, OverlayTrigger, Tooltip as BootstrapTooltip } from "react-bootstrap";
import { useDropzone } from "react-dropzone";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import "./ChurnPrediction.css";

const COLORS = ["#00C49F", "#FF4C4C"]; // Green for Retained, Red for Likely to Churn

const ChurnPrediction = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    setFile(selectedFile);
    setFileName(selectedFile?.name || "");
    setResults([]);
    setError("");
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: ".csv" });

  const handleUpload = async () => {
    if (!file) {
      alert("📂 Please upload a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/predict-churn", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(res.data.data);
      setError("");
    } catch (err) {
      setError(err.response?.data?.error || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  // Aggregate churn counts for Pie Chart
  const churnSummary = [
    { name: "Retained", value: results.filter(r => r.ChurnLabel.includes("🟢")).length },
    { name: "Likely to Churn", value: results.filter(r => r.ChurnLabel.includes("🔴")).length },
  ];

  return (
    <Container className="forecast-container my-5">
      <Card className="forecast-card p-4 shadow-lg">
        <h2 className="forecast-title mb-3">📉 Customer Churn Prediction Dashboard</h2>
        <p className="forecast-subtitle text-muted">
          Upload your customer CSV file to predict churn probability.
        </p>

        <Form className="mb-4">
          <div {...getRootProps()} className={`upload-box ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="upload-icon">⬆️</div>
            <h5>Upload CSV File</h5>
            <p>Drag & drop or click to browse</p>
            {fileName && <small className="text-muted">Selected File: {fileName}</small>}

            {/* ✅ Supported Columns Info */}
            <small className="text-muted d-block mt-2">
              <OverlayTrigger
                placement="right"
                overlay={<BootstrapTooltip>CSV must include these columns</BootstrapTooltip>}
              >
                <span style={{ cursor: "help", color: "#0d6efd" }}>ⓘ</span>
              </OverlayTrigger>{" "}
              Supported Columns: <strong>Customer / CustomerID, Gender, Age, Tenure, MonthlyCharges, TotalCharges</strong>
            </small>
          </div>

          <Button variant="dark" className="mt-3" onClick={handleUpload} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : "Upload & Predict"}
          </Button>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}
        {!results.length && !loading && <Alert variant="info">ℹ️ Upload a CSV to start churn prediction.</Alert>}

        {results.length > 0 && (
          <>
            <h5 className="text-white mt-4">📊 Churn Overview</h5>
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={churnSummary}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {churnSummary.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value} Customers`} />
                  <Legend verticalAlign="bottom" />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <h5 className="text-white mt-4">🔍 Prediction Details</h5>
            <div className="table-responsive">
              <Table striped bordered hover responsive>
                <thead className="table-dark">
                  <tr>
                    <th>Customer</th>
                    <th>Gender</th>
                    <th>Age</th>
                    <th>Tenure</th>
                    <th>Monthly Charges</th>
                    <th>Total Charges</th>
                    <th>Churn %</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.Customer}</td>
                      <td>{row.Gender === 0 ? "Female" : "Male"}</td>
                      <td>{row.Age}</td>
                      <td>{row.Tenure}</td>
                      <td>${Number(row.MonthlyCharges).toFixed(2)}</td>
                      <td>${Number(row.TotalCharges).toFixed(2)}</td>
                      <td>{row.ChurnProbability}%</td>
                      <td>{row.ChurnLabel}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          </>
        )}
      </Card>
    </Container>
  );
};

export default ChurnPrediction;
