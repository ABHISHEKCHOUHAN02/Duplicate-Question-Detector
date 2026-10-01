import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getClusterDetail } from "../api/Client.js";

export default function ClusterDetail() {
  const { clusterId } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);
    setDetail(null);
    getClusterDetail(clusterId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [clusterId]);

  return (
    <div className="cluster-detail">
      <button className="back-button" onClick={() => navigate("/clusters")}>
        ← Back to clusters
      </button>

      {loading && <p>Loading...</p>}
      {error && <p className="error-text">Failed to load cluster: {error}</p>}

      {detail && (
        <>
          <h2>
            Cluster #{detail.cluster_id} ({detail.size} questions)
          </h2>
          <ul className="question-list">
            {detail.questions.map((q) => (
              <li key={q.id}>{q.text}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}