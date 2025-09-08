import React, { useState } from "react";
import axios from "axios";
import "./RecommendationSystem.css"; // Keep your existing CSS

const RecommendationSystem = () => {
  const [file, setFile] = useState(null);
  const [customerId, setCustomerId] = useState("");
  const [expectedValue, setExpectedValue] = useState(""); // NEW
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
      setErrorMsg("⚠️ Please upload a file and enter a Customer ID.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setRecommendations([]);
    setCluster(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("customer_id", customerId);
    formData.append("expected_value", expectedValue); // NEW

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload_and_recommend",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setRecommendations(response.data.recommendations || []);
      setCluster(response.data.cluster);
    } catch (err) {
      const detail =
        err.response?.data?.detail || "🚨 Could not process recommendations.";
      setErrorMsg(detail);
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    container: {
      backgroundColor: "#595954",
      padding: "2rem 1rem",
      display: "flex",
      justifyContent: "center",
      minHeight: "100vh",
    },
    card: {
      backgroundColor: "rgba(46,46,35,0.95)",
      borderRadius: "16px",
      padding: "2rem",
      boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
      width: "100%",
      maxWidth: "1000px",
      color: "#fff",
      transition: "all 0.3s ease",
      boxSizing: "border-box",
    },
    header: {
      textAlign: "center",
      marginBottom: "2rem",
      fontSize: "2rem",
      letterSpacing: "0.5px",
    },
    uploadBox: {
      border: "2px dashed #888",
      borderRadius: "12px",
      padding: "1.5rem",
      textAlign: "center",
      cursor: "pointer",
      backgroundColor: "#3a3a2f",
      transition: "all 0.3s ease",
      wordBreak: "break-word",
    },
    formGroup: {
      marginTop: "1.5rem",
      marginBottom: "1rem",
    },
    label: {
      display: "block",
      marginBottom: "0.5rem",
      fontWeight: "bold",
      color: "#fff",
    },
    inputText: {
      width: "100%",
      padding: "0.7rem",
      borderRadius: "8px",
      border: "1px solid #555",
      backgroundColor: "#2e2e23",
      color: "#fff",
      fontSize: "1rem",
    },
    button: {
      marginTop: "1.5rem",
      padding: "0.8rem 1.5rem",
      backgroundColor: "#007bff",
      color: "#fff",
      border: "none",
      borderRadius: "8px",
      fontWeight: "bold",
      width: "100%",
      fontSize: "1.1rem",
      transition: "all 0.2s ease",
    },
    error: {
      color: "#ff6b6b",
      marginTop: "1rem",
      fontWeight: "bold",
      textAlign: "center",
    },
    results: {
      marginTop: "2rem",
      backgroundColor: "rgba(70,70,60,0.5)",
      borderRadius: "12px",
      padding: "1.5rem",
    },
    resultsHeader: {
      color: "#fff",
      marginBottom: "1rem",
      fontSize: "1.5rem",
    },
    list: {
      paddingLeft: "1.2rem",
    },
    listItem: {
      marginBottom: "0.6rem",
      fontSize: "1.1rem",
      color: "#fff",
      transition: "all 0.2s ease",
    },
    noResults: {
      marginTop: "1rem",
      color: "#ccc",
      fontStyle: "italic",
      textAlign: "center",
    },
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h2 style={styles.header}>🛍️ Smart Product Recommender</h2>

        {/* Upload CSV Box */}
        <div
          className={`upload-box ${file ? "active" : ""}`}
          onClick={() => document.getElementById("fileInput").click()}
          style={styles.uploadBox}
        >
          <input
            type="file"
            id="fileInput"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ display: "none" }}
          />
          <div className="upload-icon">⬆️</div>
          <h5>{file ? "File Selected" : "Upload CSV File"}</h5>
          <p>{file ? file.name : "Drag & drop or click to browse"}</p>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Enter Customer ID:</label>
          <input
            type="text"
            value={customerId}
            placeholder="e.g. 3"
            onChange={(e) => setCustomerId(e.target.value)}
            style={styles.inputText}
          />
        </div>

        {/* NEW Expected Column Value */}
        <div style={styles.formGroup}>
          <label style={styles.label}>Expected Column Value:</label>
          <input
            type="text"
            value={expectedValue}
            placeholder="e.g. 1000"
            onChange={(e) => setExpectedValue(e.target.value)}
            style={styles.inputText}
          />
        </div>

        <button
          onClick={handleUpload}
          style={{
            ...styles.button,
            opacity: loading ? 0.7 : 1,
            cursor: loading ? "not-allowed" : "pointer",
          }}
          disabled={loading}
        >
          {loading ? "⏳ Processing..." : "🔍 Get Recommendations"}
        </button>

        {errorMsg && <p style={styles.error}>{errorMsg}</p>}

        {recommendations.length > 0 && (
          <div style={styles.results}>
            <h3 style={styles.resultsHeader}>
              🎯 Recommendations for <strong>{customerId}</strong>
              <br />
              Cluster:{" "}
              <strong>{clusterLabels[cluster] ?? `Cluster ${cluster ?? "-"}`}</strong>
            </h3>
            <ul style={styles.list}>
              {recommendations.map((item, index) => (
                <li key={index} style={styles.listItem}>
                  ⭐ {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {!loading && recommendations.length === 0 && cluster && (
          <p style={styles.noResults}>
            🙁 No personalized products found for this customer.
          </p>
        )}
      </div>
    </div>
  );
};

export default RecommendationSystem;
