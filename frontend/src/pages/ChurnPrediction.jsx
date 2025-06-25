import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const ChurnPrediction = () => {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState('random_forest');
  const [report, setReport] = useState(null);
  const [matrix, setMatrix] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please upload a CSV file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post(
        `http://localhost:8000/churn-prediction?model=${model}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      setReport(response.data.classification_report);
      setMatrix(response.data.confusion_matrix);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-4">
      <h2>📉 Churn Prediction</h2>

      <div className="row mb-3">
        <div className="col-md-6">
          <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
        </div>
        <div className="col-md-4">
          <select className="form-select" value={model} onChange={handleModelChange}>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>
        <div className="col-md-2">
          <button className="btn btn-primary w-100" onClick={handleUpload} disabled={loading}>
            {loading ? 'Processing...' : 'Predict'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {report && (
        <div className="mt-4">
          <h5>📋 Classification Report</h5>
          <table className="table table-bordered table-hover">
            <thead className="table-light">
              <tr>
                <th>Label</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(report)
                .filter(([key]) => key !== 'accuracy')
                .map(([label, metrics], idx) => (
                  <tr key={idx}>
                    <td>{label}</td>
                    <td>{metrics.precision?.toFixed(2) ?? '-'}</td>
                    <td>{metrics.recall?.toFixed(2) ?? '-'}</td>
                    <td>{metrics['f1-score']?.toFixed(2) ?? '-'}</td>
                    <td>{metrics.support ?? '-'}</td>
                  </tr>
                ))}
              {'accuracy' in report && (
                <tr className="table-info">
                  <td colSpan="5">
                    <strong>Overall Accuracy: {report.accuracy.toFixed(2)}</strong>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {matrix && (
        <div className="mt-4">
          <h5>📊 Confusion Matrix</h5>
          <table className="table table-bordered text-center">
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  {row.map((val, j) => (
                    <td key={j}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ChurnPrediction;
