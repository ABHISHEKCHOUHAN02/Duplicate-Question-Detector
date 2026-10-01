import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getClusters } from "../api/Client.js";

export default function ClusterList() {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);
    getClusters()
      .then((data) => {
        if (!cancelled) setClusters(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true; // avoids setting state after this component has unmounted
    };
  }, []);

  if (loading) return <p>Loading clusters...</p>;
  if (error) return <p className="error-text">Failed to load clusters: {error}</p>;
  if (clusters.length === 0) {
    return <p>No duplicate clusters yet — submit a few similar questions first.</p>;
  }

  return (
    <div className="cluster-list">
      {clusters.map((c) => (
        <button
          key={c.cluster_id}
          className="cluster-card"
          onClick={() => navigate(`/clusters/${c.cluster_id}`)}
        >
          <span className="cluster-id">Cluster #{c.cluster_id}</span>
          <span className="cluster-size">{c.size} questions</span>
        </button>
      ))}
    </div>
  );
}