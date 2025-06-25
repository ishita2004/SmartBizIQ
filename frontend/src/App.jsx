import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Layout from './components/Layout';  // ✅ new layout
import Navigation from './pages/Navigation';
import SalesForecasting from './pages/SalesForecast';
import CustomerSegmentation from './pages/CustomerSegmentation';
import ChurnPrediction from './pages/ChurnPrediction';
import AnomalyDetection from './pages/AnomalyDetection';
import RecommendationSystem from './pages/RecommendationSystem';

function App() {
  return (
    <Router>
      <Routes>
        {/* Wrap all routes inside Layout for navbar */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigation />} /> {/* default home route */}
          <Route path="sales-forecasting" element={<SalesForecasting />} />
          <Route path="customer-segmentation" element={<CustomerSegmentation />} />
          <Route path="churn-prediction" element={<ChurnPrediction />} />
          <Route path="anomaly-detection" element={<AnomalyDetection />} />
          <Route path="recommendation-system" element={<RecommendationSystem />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
