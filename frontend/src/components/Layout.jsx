import React, { useEffect, useState } from 'react';
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { FaChartLine, FaUsers, FaExclamationTriangle, FaRobot, FaBrain, FaBars, FaTimes } from "react-icons/fa";
import './Layout.css';

const Layout = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loaded, setLoaded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const hideNavbarOn = ['/login', '/signup'];

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  const isLoggedIn = !!localStorage.getItem('token');
  const showNavbar = !hideNavbarOn.includes(location.pathname);

  useEffect(() => {
    setLoaded(true);
  }, []);

  const toggleMenu = () => setMenuOpen(!menuOpen);

  return (
    <div className="layout-container">
      {showNavbar && (
        <nav className="navbar">
          <h3 className="logo">SmartBizIQ</h3>
          <div className={`links ${menuOpen ? 'open' : ''}`}>
            <Link to="/" onClick={() => setMenuOpen(false)}>Home</Link>
            <Link to="/sales-forecasting" onClick={() => setMenuOpen(false)}>Sales Forecasting</Link>
            <Link to="/customer-segmentation" onClick={() => setMenuOpen(false)}>Customer Segmentation</Link>
            <Link to="/churn-prediction" onClick={() => setMenuOpen(false)}>Churn Prediction</Link>
            <Link to="/anomaly-detection" onClick={() => setMenuOpen(false)}>Anomaly Detection</Link>
            <Link to="/recommendation-system" onClick={() => setMenuOpen(false)}>Recommendations</Link>
            <Link to="/dashboard" onClick={() => setMenuOpen(false)}>BizzBOT</Link>
            {isLoggedIn && (
              <button onClick={handleLogout} className="logout">Logout</button>
            )}
          </div>
          <div className="hamburger" onClick={toggleMenu}>
            {menuOpen ? <FaTimes /> : <FaBars />}
          </div>
        </nav>
      )}

      <main>
        {location.pathname === '/' ? (
          <div style={{ opacity: loaded ? 1 : 0, transition: 'opacity 1s ease-in' }}>
            <section className="hero">
              <h1>Welcome to SmartBizIQ</h1>
              <p>Analyze, Predict, and Grow Your Business with Intelligent Insights</p>
            </section>

            <section className="features">
              <h2>Explore Our Key Features</h2>
              <p className="features-subtitle">
                Unlock insights and make smarter decisions with our powerful tools.
              </p>
              <div className="feature-grid">
                {[
                  { name: "Sales Forecasting", icon: <FaChartLine /> },
                  { name: "Customer Segmentation", icon: <FaUsers /> },
                  { name: "Churn Prediction", icon: <FaExclamationTriangle /> },
                  { name: "Anomaly Detection", icon: <FaRobot /> },
                  { name: "Recommendation System", icon: <FaBrain /> },
                ].map((feature, idx) => (
                  <div key={idx} className="feature-card">
                    <div className="feature-icon">{feature.icon}</div>
                    <h3>{feature.name}</h3>
                    <p>Enhance your business decisions using {feature.name.toLowerCase()}.</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        ) : (
          <Outlet />
        )}
      </main>
    </div>
  );
};

export default Layout;
