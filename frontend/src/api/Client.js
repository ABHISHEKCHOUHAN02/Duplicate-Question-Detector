const BASE_URL = "http://127.0.0.1:8000";

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = res.statusText;
    }
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  return res.json();
}

export function submitQuestion(text) {
  return request("/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function getQuestion(id) {
  return request(`/questions/${id}`);
}

export function getDuplicates(id) {
  return request(`/questions/${id}/duplicates`);
}

export function getClusters(minSize = 2) {
  return request(`/clusters?min_size=${minSize}`);
}

export function getClusterDetail(clusterId) {
  return request(`/clusters/${clusterId}`);
}