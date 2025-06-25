import React, { useState } from 'react';
import axios from 'axios';

function AnomalyDetection() {
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState("isolation_forest");
  const [result, setResult] = useState(null);
  const [imageSrc, setImageSrc] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleMethodChange = (e) => {
    setMethod(e.target.value);
  };

  const handleSubmit = async () => {
    if (!file) return alert("Please select a CSV file.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `http://localhost:8000/anomaly-detection?method=${method}`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(response.data.data);
      setImageSrc("data:image/png;base64," + response.data.plot);
    } catch (error) {
      alert("An error occurred: " + error.response?.data?.error || error.message);
    }
  };

  return (
    <div className="container mt-4">
      <h2>🔎 Anomaly Detection</h2>
      <div className="form-group">
        <label>📂 Choose CSV File:</label>
        <input type="file" accept=".csv" onChange={handleFileChange} className="form-control" />
      </div>

      <div className="form-group mt-2">
        <label>🧠 Select Detection Method:</label>
        <select className="form-control" onChange={handleMethodChange} value={method}>
          <option value="isolation_forest">Isolation Forest</option>
          <option value="svm">One-Class SVM</option>
          <option value="zscore">Z-Score</option>
        </select>
      </div>

      <button className="btn btn-primary mt-3" onClick={handleSubmit}>Detect Anomalies</button>

      {imageSrc && (
        <div className="mt-4">
          <h4>📊 Anomaly Plot</h4>
          <img src={imageSrc} alt="Anomaly Plot" className="img-fluid" />
        </div>
      )}

      {result && (
        <div className="mt-4">
          <h4>📋 Anomaly Table</h4>
          <div className="table-responsive">
            <table className="table table-bordered table-striped">
              <thead>
                <tr>
                  {Object.keys(result[0]).map((col, idx) => (
                    <th key={idx}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((val, j) => (
                      <td key={j}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default AnomalyDetection;
