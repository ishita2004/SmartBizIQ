import React from 'react';
import { Link, Outlet } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Layout = () => {
  return (
    <>
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
        <div className="container-fluid">
          <Link className="navbar-brand" to="/">SmartBizIQ</Link>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon" />
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto">
              <li className="nav-item">
                <Link className="nav-link" to="/sales-forecasting">Sales Forecasting</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/customer-segmentation">Customer Segmentation</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/churn-prediction">Churn Prediction</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/anomaly-detection">Anomaly Detection</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/recommendation-system">Recommendation System</Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      {/* Render child pages inside Layout */}
      <main className="container mt-4">
        <Outlet />
      </main>
    </>
  );
};

export default Layout;
