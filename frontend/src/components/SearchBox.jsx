import { useState } from "react";
import { submitQuestion } from "../api/Client.js";

export default function SearchBox({ onResult }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await submitQuestion(trimmed);
      onResult(result);
      setText("");
    } catch (err) {
      setError(err.message);
      onResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="search-box" onSubmit={handleSubmit}>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. How do I learn Python?"
        disabled={loading}
      />
      <button type="submit" disabled={loading}>
        {loading ? "Checking..." : "Check for duplicates"}
      </button>
      {error && <p className="error-text">{error}</p>}
    </form>
  );
}