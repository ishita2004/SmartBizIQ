import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const CustomerSegmentation = () => {
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("kmeans");
  const [results, setResults] = useState([]);
  const [plot, setPlot] = useState(null);
  const [summaries, setSummaries] = useState({});
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const handleMethodChange = (e) => {
    setMethod(e.target.value);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please upload a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `http://localhost:8000/segmentation/segment-customers?method=${method}`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      setResults(response.data.data);
      setPlot(response.data.plot);
      setSummaries(response.data.summaries || {});
    } catch (err) {
      setError(err.response?.data?.error || "Upload failed.");
    }
  };

  return (
    <div className="container mt-4">
      <h2 className="mb-4">📊 Customer Segmentation</h2>

      <div className="row mb-3">
        <div className="col-md-6">
          <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
        </div>

        <div className="col-md-4">
          <select className="form-select" value={method} onChange={handleMethodChange}>
            <option value="kmeans">KMeans</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </div>

        <div className="col-md-2">
          <button className="btn btn-primary w-100" onClick={handleUpload}>Segment</button>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {results.length > 0 && (
        <div className="mt-4">
          <h5>📋 Segmentation Results</h5>
          <div className="table-responsive">
            <table className="table table-bordered table-hover">
              <thead className="table-light">
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
                    <td>{row.Annual_Income}</td>
                    <td>{row.Spending_Score}</td>
                    <td>{row.Cluster}</td>
                    <td>{row.Label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {Object.keys(summaries).length > 0 && (
        <div className="mt-4">
          <h5>🧠 Segment Summaries</h5>
          <ul className="list-group">
            {Object.entries(summaries).map(([label, summary], idx) => (
              <li key={idx} className="list-group-item">
                <strong>{label}:</strong> {summary}
              </li>
            ))}
          </ul>
        </div>
      )}

      {plot && (
        <div className="mt-4">
          <h5>📈 Cluster Visualization</h5>
          <img src={`data:image/png;base64,${plot}`} alt="Segmentation Plot" className="img-fluid border" />
        </div>
      )}
    </div>
  );
};

export default CustomerSegmentation;
