import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import ComparisonLab from './pages/ComparisonLab';

import VisualizationLab from './pages/VisualizationLab';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/comparison" element={<ComparisonLab />} />
        <Route path="/visualization" element={<VisualizationLab />} />
      </Routes>
    </Router>
  );
}

export default App;
