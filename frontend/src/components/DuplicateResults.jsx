export default function DuplicateResults({ result }) {
  if (!result) return null;

  const { text, cluster_id, is_new_cluster, duplicates } = result;

  return (
    <div className="results">
      <p className="submitted-text">
        You asked: <strong>&ldquo;{text}&rdquo;</strong>
      </p>

      {cluster_id === null && (
        <p className="status status-none">
          No existing duplicates found. This question has been saved.
        </p>
      )}

      {cluster_id !== null && is_new_cluster && (
        <p className="status status-new">
          Found a duplicate — a new cluster was created.
        </p>
      )}

      {cluster_id !== null && !is_new_cluster && (
        <p className="status status-joined">
          This question joined an existing cluster of duplicates.
        </p>
      )}

      {duplicates.length > 0 && (
        <ul className="duplicate-list">
          {duplicates.map((d) => (
            <li key={d.id}>
              <span className="duplicate-text">{d.text}</span>
              <span className="similarity-badge">
                {(d.similarity * 100).toFixed(1)}% match
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}