import React, { useState } from 'react';
import axios from 'axios';
import { Container, Card, Form, Button, Alert, Table, Spinner } from 'react-bootstrap';

const AnomalyDetection = () => {
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("isolation_forest");
  const [filename, setFilename] = useState("");
  const [results, setResults] = useState([]);
  const [plot, setPlot] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const methodDescriptions = {
    isolation_forest: "Uses decision trees to isolate anomalies quickly. Best for high-dimensional data.",
    svm: "One-Class Support Vector Machine. Good for data with tight boundaries.",
    zscore: "Identifies outliers using standard deviation thresholds. Simple but effective for numeric data."
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setFilename(selected?.name || "");
    setResults([]);
    setPlot("");
    setError("");
  };

  const handleReset = () => {
    setFile(null);
    setFilename("");
    setResults([]);
    setPlot("");
    setError("");
    setMethod("isolation_forest");
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("📂 Please upload a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await axios.post(`http://localhost:8000/anomaly-detection?method=${method}`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResults(res.data.data || []);
      setPlot("data:image/png;base64," + res.data.plot);
      setError("");
    } catch (err) {
      setError(err.response?.data?.error || "Detection failed. Please check your CSV format.");
    } finally {
      setLoading(false);
    }
  };

  const total = results.length;
  const anomalies = results.filter((row) => row.Anomaly === 1).length;
  const anomalyPercent = total > 0 ? ((anomalies / total) * 100).toFixed(2) : 0;

  return (
    <Container className="my-5">
      <Card className="p-4 shadow-sm">
        <h2 className="mb-3">🔍 Anomaly Detection</h2>
        <p className="text-muted">Upload a CSV file and select an algorithm to identify unusual patterns or outliers in your data.</p>

        <Form className="mb-4">
          <Form.Group className="mb-3">
            <Form.Label>📂 Upload CSV File</Form.Label>
            <Form.Control type="file" accept=".csv" onChange={handleFileChange} />
            {filename && <small className="text-muted">Selected File: {filename}</small>}
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>🧠 Detection Method</Form.Label>
            <Form.Select value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="isolation_forest">Isolation Forest</option>
              <option value="svm">One-Class SVM</option>
              <option value="zscore">Z-Score</option>
            </Form.Select>
            <Form.Text className="text-muted">{methodDescriptions[method]}</Form.Text>
          </Form.Group>

          <div className="d-flex gap-2">
            <Button variant="dark" onClick={handleSubmit} disabled={loading}>
              {loading ? <Spinner animation="border" size="sm" /> : "🚀 Detect Anomalies"}
            </Button>
            <Button variant="outline-secondary" onClick={handleReset}>🔁 Reset</Button>
            <a href="/sample.csv" className="btn btn-outline-info" download>⬇️ Sample CSV</a>
          </div>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}

        {plot && (
          <div className="mt-4">
            <h5>📈 Anomaly Plot</h5>
            <img src={plot} alt="Anomaly Plot" className="img-fluid border rounded" />
          </div>
        )}

        {results.length > 0 && (
          <div className="mt-4">
            <h5>📋 Detection Results</h5>
            <p><strong>Total Rows:</strong> {total}</p>
            <p><strong>Anomalies Detected:</strong> {anomalies} ({anomalyPercent}%)</p>

            <div className="table-responsive">
              <Table striped bordered hover>
                <thead className="table-dark">
                  <tr>
                    {Object.keys(results[0]).map((col, idx) => (
                      <th key={idx}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, idx) => (
                    <tr key={idx} className={row.Anomaly === 1 ? 'table-danger' : ''}>
                      {Object.values(row).map((val, i) => (
                        <td key={i}>{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          </div>
        )}
      </Card>
    </Container>
  );
};

export default AnomalyDetection;
