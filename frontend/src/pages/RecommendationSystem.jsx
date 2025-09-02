import React, { useState } from "react";
import axios from "axios";

const RecommendationSystem = () => {
  const [file, setFile] = useState(null);
  const [customerId, setCustomerId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [cluster, setCluster] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const clusterLabels = {
    0: "🛍️ Budget Conscious",
    1: "💰 High-Spender",
    2: "🧍 Casual Buyer",
    3: "🎯 Target Shopper",
    4: "📦 Bulk Buyer",
  };

  const handleUpload = async () => {
    if (!file || !customerId) {
      alert("Please upload a file and enter a Customer ID.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setRecommendations([]);
    setCluster(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post("http://localhost:5000/upload", formData);

      const recommendData = new FormData();
      recommendData.append("customer_id", customerId);

      const response = await axios.post("http://localhost:5000/recommend", recommendData);

      setRecommendations(response.data.recommendations || []);
      setCluster(response.data.cluster);
    } catch (err) {
      console.error(err);
      setErrorMsg("🚨 Could not process recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.header}>🛍️ Smart Product Recommender</h2>

        <div style={styles.formGroup}>
          <label style={styles.label}>Upload CSV:</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            style={styles.inputFile}
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Enter Customer ID:</label>
          <input
            type="text"
            value={customerId}
            placeholder="e.g. C12345"
            onChange={(e) => setCustomerId(e.target.value)}
            style={styles.inputText}
          />
        </div>

        <button onClick={handleUpload} style={styles.button}>
          {loading ? "⏳ Processing..." : "🔍 Get Recommendations"}
        </button>

        {errorMsg && <p style={styles.error}>{errorMsg}</p>}

        {recommendations.length > 0 && (
          <div style={styles.results}>
            <h3>
              🎯 Recommendations for <strong>{customerId}</strong><br />
              Cluster: <strong>{clusterLabels[cluster] || cluster}</strong>
            </h3>
            <ul style={styles.list}>
              {recommendations.map((item, index) => (
                <li key={index} style={styles.listItem}>⭐ {item}</li>
              ))}
            </ul>
          </div>
        )}

        {!loading && recommendations.length === 0 && cluster && (
          <p style={styles.noResults}>🙁 No personalized products found for this customer.</p>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: "#f7f9fc",
    padding: "3rem",
    display: "flex",
    justifyContent: "center",
    minHeight: "100vh",
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: "10px",
    padding: "2rem",
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    width: "100%",
    maxWidth: "600px",
  },
  header: {
    textAlign: "center",
    marginBottom: "1.5rem",
  },
  formGroup: {
    marginBottom: "1rem",
  },
  label: {
    display: "block",
    marginBottom: "0.5rem",
    fontWeight: "bold",
  },
  inputFile: {
    width: "100%",
  },
  inputText: {
    width: "100%",
    padding: "0.5rem",
    borderRadius: "6px",
    border: "1px solid #ccc",
  },
  button: {
    marginTop: "1rem",
    padding: "0.6rem 1.2rem",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "bold",
    width: "100%",
  },
  error: {
    color: "red",
    marginTop: "1rem",
  },
  results: {
    marginTop: "2rem",
  },
  list: {
    paddingLeft: "1.2rem",
  },
  listItem: {
    marginBottom: "0.5rem",
    fontSize: "1rem",
  },
  noResults: {
    marginTop: "1rem",
    color: "#666",
    fontStyle: "italic",
  },
};

export default RecommendationSystem;
