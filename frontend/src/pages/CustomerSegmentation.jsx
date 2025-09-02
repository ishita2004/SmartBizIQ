import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Card,
  Form,
  Button,
  Row,
  Col,
  Alert,
  Table,
  Spinner,
  OverlayTrigger,
  Tooltip as BootstrapTooltip,
  ListGroup
} from "react-bootstrap";

const CustomerSegmentation = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [method, setMethod] = useState("kmeans");
  const [results, setResults] = useState([]);
  const [plot, setPlot] = useState(null);
  const [summaries, setSummaries] = useState({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setFileName(selected?.name || "");
    setError("");
    setResults([]);
    setSummaries({});
    setPlot(null);
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
        `http://localhost:8000/segmentation/segment-customers?method=${method}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResults(res.data.data);
      setPlot(res.data.plot);
      setSummaries(res.data.summaries || {});
      setError("");
    } catch (err) {
      console.error("Upload failed:", err);
      setError(err.response?.data?.error || "❌ Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="my-5">
      <Card className="shadow-lg border-0 p-4">
        <h2 className="mb-4" style={{ color: "#000" }}>👥 Customer Segmentation Dashboard</h2>
        <p className="text-muted">Upload your customer data (CSV) and segment them using clustering models like KMeans or DBSCAN. Required columns: <code>Age, Annual_Income, Spending_Score</code>.</p>

        <Form className="mb-4">
          <Form.Group className="mb-3">
            <Form.Label className="text-dark">
              🧠 Choose Clustering Method{" "}
              <OverlayTrigger placement="right" overlay={<BootstrapTooltip>Choose between KMeans (popular for structured clusters) and DBSCAN (for density-based groups and outlier detection)</BootstrapTooltip>}>
                <span style={{ cursor: "help", color: "#0d6efd" }}>ⓘ</span>
              </OverlayTrigger>
            </Form.Label>
            <Form.Select value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="kmeans">KMeans</option>
              <option value="dbscan">DBSCAN</option>
            </Form.Select>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label className="text-dark">📂 Upload Customer CSV File</Form.Label>
            <Form.Control type="file" accept=".csv" onChange={handleFileChange} />
            {fileName && <small className="text-muted">Selected File: {fileName}</small>}
          </Form.Group>

          <Button variant="dark" onClick={handleUpload} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : "Upload & Segment"}
          </Button>
        </Form>

        {error && <Alert variant="danger">{error}</Alert>}
        {!results.length && !loading && <Alert variant="info">ℹ️ Upload a CSV and choose a clustering method to start segmentation.</Alert>}

        {results.length > 0 && (
          <>
            <h4 className="text-dark mb-3">📋 Segmentation Results</h4>
            <div className="table-responsive">
              <Table striped bordered hover responsive className="text-dark">
                <thead className="table-dark">
                  <tr>
                    <th>Age</th>
                    <th>Annual Income</th>
                    <th>Spending Score</th>
                    <th>Cluster</th>
                    <th>Label</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.Age}</td>
                      <td>₹{row.Annual_Income.toLocaleString()}</td>
                      <td>{row.Spending_Score}</td>
                      <td>{row.Cluster}</td>
                      <td>{row.Label}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>

            <Card className="mt-4 p-3 bg-light">
              <h5 className="text-dark">📊 Cluster Visualization</h5>
              {plot && (
                <img
                  src={`data:image/png;base64,${plot}`}
                  alt="Segmentation Plot"
                  className="img-fluid border mt-3"
                />
              )}
            </Card>

            {Object.keys(summaries).length > 0 && (
              <Card className="mt-4 p-3 bg-light border-info">
                <h5 className="text-dark">🧠 Segment Summaries</h5>
                <ListGroup variant="flush">
                  {Object.entries(summaries).map(([label, summary], idx) => (
                    <ListGroup.Item key={idx}>
                      <strong>{label}:</strong> {summary}
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              </Card>
            )}
          </>
        )}
      </Card>
    </Container>
  );
};

export default CustomerSegmentation;
