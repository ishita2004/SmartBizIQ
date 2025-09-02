import React from 'react';
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';

const Layout = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const hideNavbarOn = ['/', '/login', '/signup'];

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  const isLoggedIn = !!localStorage.getItem('token');
  const showNavbar = !hideNavbarOn.includes(location.pathname);

  return (
    <div>
      {showNavbar && (
        <nav style={styles.navbar}>
          <h3 style={styles.logo}>SmartBizIQ</h3>
          <div style={styles.links}>
            <Link to="/" style={styles.link}>Home</Link>
            <Link to="/sales-forecasting" style={styles.link}>Sales Forecasting</Link>
            <Link to="/customer-segmentation" style={styles.link}>Customer Segmentation</Link>
            <Link to="/churn-prediction" style={styles.link}>Churn Prediction</Link>
            <Link to="/anomaly-detection" style={styles.link}>Anomaly Detection</Link>
            <Link to="/recommendation-system" style={styles.link}>Recommendations</Link>
            <Link to="/dashboard" style={styles.link}>Dashboard</Link>
            {isLoggedIn && (
              <button onClick={handleLogout} style={styles.logout}>Logout</button>
            )}
          </div>
        </nav>
      )}
      <main style={{ padding: '20px' }}>
        <Outlet />
      </main>
    </div>
  );
};

const styles = {
  navbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    background: '#333',
    padding: '10px 20px',
    color: 'white',
  },
  logo: {
    margin: 0,
  },
  links: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
  },
  link: {
    color: 'white',
    textDecoration: 'none',
  },
  logout: {
    background: '#ff4d4d',
    color: 'white',
    border: 'none',
    padding: '5px 10px',
    cursor: 'pointer',
    borderRadius: '4px',
  },
};

export default Layout;
