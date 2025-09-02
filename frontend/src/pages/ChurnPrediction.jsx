import React, { useState } from "react";
import axios from "axios";
import { Container, Card, Form, Button, Table, Alert, Spinner } from "react-bootstrap";

const ChurnPrediction = () => {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setFilename(selected?.name || "");
    setResults([]);
    setError("");
  };

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

  return (
    <Container className="my-5">
      <Card className="p-4 shadow-sm">
        <h2 className="mb-3">📉 Customer Churn Prediction</h2>
        <p className="text-muted">
          Upload a CSV file with customer details to predict churn probability.
        </p>

        <Form className="mb-4">
          <Form.Group controlId="formFile" className="mb-3">
            <Form.Label>📂 Upload Customer CSV</Form.Label>
            <Form.Control type="file" accept=".csv" onChange={handleFileChange} />
            {filename && <small className="text-muted">Selected: {filename}</small>}
          </Form.Group>

          <Button variant="dark" onClick={handleUpload} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : "Upload & Predict"}
          </Button>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}

        {results.length > 0 && (
          <>
            <h5>🔍 Prediction Results</h5>
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
                    <td>${row.MonthlyCharges}</td>
                    <td>${row.TotalCharges}</td>
                    <td>{row.ChurnProbability}%</td>
                    <td>{row.ChurnLabel}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </>
        )}
      </Card>
    </Container>
  );
};

export default ChurnPrediction;
