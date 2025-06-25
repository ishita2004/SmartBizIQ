import React, { useState } from 'react';
import axios from 'axios';

const RecommendationSystem = () => {
  const [file, setFile] = useState(null);
  const [customerId, setCustomerId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file || !customerId) {
      setError("Upload a CSV and enter a Customer ID.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        `http://localhost:8000/recommendation?customer_id=${customerId}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setRecommendations(response.data.recommendations || []);
      setError('');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch recommendations.');
    }
  };

  return (
    <div className="container mt-4">
      <h2>🎁 Recommendation System</h2>

      <div className="row mb-3">
        <div className="col-md-4">
          <input type="file" className="form-control" onChange={(e) => setFile(e.target.files[0])} />
        </div>
        <div className="col-md-4">
          <input type="number" className="form-control" placeholder="Customer ID" onChange={(e) => setCustomerId(e.target.value)} />
        </div>
        <div className="col-md-4">
          <button className="btn btn-primary w-100" onClick={handleUpload}>Recommend</button>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {recommendations.length > 0 && (
        <div className="mt-4">
          <h5>🧠 Recommended Products</h5>
          <ul className="list-group">
            {recommendations.map((rec, idx) => (
              <li key={idx} className="list-group-item">
                {rec.Product} ({rec.Category})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default RecommendationSystem;
