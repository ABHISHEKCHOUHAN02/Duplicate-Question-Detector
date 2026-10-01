import { useState } from "react";
import { NavLink, Routes, Route, Navigate } from "react-router-dom";
import SearchBox from "./components/SearchBox.jsx";
import DuplicateResults from "./components/DuplicateResults.jsx";
import ClusterList from "./components/ClusterList.jsx";
import ClusterDetail from "./components/ClusterDetail.jsx";

function SubmitPage() {
  const [submitResult, setSubmitResult] = useState(null);
  return (
    <>
      <SearchBox onResult={setSubmitResult} />
      {submitResult && <DuplicateResults result={submitResult} />}
    </>
  );
}

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Duplicate Question Detector</h1>
        <nav className="tabs">
          <NavLink
            to="/"
            end
            className={({ isActive }) => (isActive ? "tab active" : "tab")}
          >
            Submit a Question
          </NavLink>
          <NavLink
            to="/clusters"
            className={({ isActive }) => (isActive ? "tab active" : "tab")}
          >
            Browse Clusters
          </NavLink>
        </nav>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<SubmitPage />} />
          <Route path="/clusters" element={<ClusterList />} />
          <Route path="/clusters/:clusterId" element={<ClusterDetail />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}