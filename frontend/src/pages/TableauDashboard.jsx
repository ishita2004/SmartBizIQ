import React from "react";

const TableauDashboard = () => {
  return (
    <div className="container mt-4">
      <h2>📊 Interactive Tableau Dashboard</h2>
      <p className="text-muted">Analyze historical trends, regional performance, and forecasts.</p>

      <div style={{ height: "700px", border: "1px solid #ccc" }}>
        <iframe
          title="Tableau Dashboard"
          width="100%"
          height="100%"
          src="https://public.tableau.com/views/YOUR_DASHBOARD_NAME/SHEET_NAME?:embed=true&:display_count=yes"
          frameBorder="0"
        ></iframe>
      </div>
    </div>
  );
};

export default TableauDashboard;
