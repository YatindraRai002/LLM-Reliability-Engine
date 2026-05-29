const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AnalyzeResult {
  prompt: string;
  result: {
    score: number;
    label: "low" | "medium" | "high";
    explanation: string;
    calibration_score: number;
    uncertainty_score: number;
    cross_check_score: number;
    weights_used: Record<string, number>;
    thresholds_used: Record<string, number>;
    n_samples_used: number;
    groq_available: boolean;
    mode: string;
  };
  calibration_detail: {
    response: string;
    token_probs: number[];
    mean_confidence: number;
    min_confidence: number;
    confidence_std: number;
    n_tokens: number;
    success: boolean;
  };
  uncertainty_detail: {
    uncertainty_score: number;
    normalized_entropy: number;
    n_semantic_clusters: number;
    mean_pairwise_similarity: number;
    cluster_labels: number[];
    embeddings_2d: number[][];
    responses: string[];
  };
  cross_check_detail: {
    local_response: string;
    groq_response: string | null;
    groq_available: boolean;
    groq_model: string;
    error: string | null;
    error_type?: string;
    verdict: string;
    cross_check_uncertainty: number;
    symmetric_agreement: number;
    ab_detail: Record<string, number>;
    ba_detail: Record<string, number>;
  };
  explanation_detail?: {
    flagged_spans: { text: string; confidence: number; severity: string }[];
    contradicting_sentences: {
      sentence: string;
      contradiction_score: number;
      entailment_score: number;
    }[];
    signal_pct: Record<string, number>;
    highlighted_html: string;
    recommendations: string[];
  };
  timings: Record<string, number>;
}

export interface HistoryRow {
  id: string;
  timestamp: string;
  prompt: string;
  score: number;
  label: string;
  cal: number;
  unc: number;
  cc: number;
  weights: string;
}

export async function analyzeQuery(query: string): Promise<AnalyzeResult> {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Analysis failed: ${detail}`);
  }
  return res.json();
}

export async function getHistory(): Promise<{
  history: HistoryRow[];
  total: number;
}> {
  const res = await fetch(`${API_BASE}/api/history`);
  return res.json();
}

export async function clearHistory(): Promise<void> {
  await fetch(`${API_BASE}/api/history`, { method: "DELETE" });
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    return data.status === "healthy";
  } catch {
    return false;
  }
}
