import React from "react";
import { Link } from "react-router-dom";

const Landing = () => {
  return (
    <div style={{ padding: "3rem", textAlign: "center" }}>
      <h1>📊 Welcome to SmartBizIQ</h1>
      <p>Empowering businesses with smart analytics & forecasting</p>
      <div style={{ marginTop: "2rem" }}>
        <Link to="/login"><button style={btn}>Login</button></Link>
        <Link to="/signup"><button style={btn}>Sign Up</button></Link>
      </div>
    </div>
  );
};

const btn = {
  margin: "0 1rem",
  padding: "0.75rem 1.5rem",
  fontSize: "16px",
  borderRadius: "8px",
  border: "none",
  cursor: "pointer",
  backgroundColor: "#007bff",
  color: "#fff"
};

export default Landing;
