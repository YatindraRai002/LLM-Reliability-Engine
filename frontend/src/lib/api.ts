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
    // Field names match core/explainer.py FlaggedSpan dataclass exactly
    flagged_spans: {
      token: string;
      position: number;
      probability: number;
      reason: string;
    }[];
    // Field names match core/explainer.py ContradictingSentence dataclass exactly
    contradicting_sentences: {
      text: string;
      nli_score: number;
      label: string;
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
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: query }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Analysis start failed: ${detail}`);
  }
  
  const { job_id } = await res.json();
  
  // Poll for result
  while (true) {
    const pollRes = await fetch(`${API_BASE}/result/${job_id}`);
    if (!pollRes.ok) {
        throw new Error(`Polling failed`);
    }
    const data = await pollRes.json();
    if (data.status === "done") {
        return data.result;
    } else if (data.status === "error") {
        throw new Error(data.error);
    }
    
    // Wait 2 seconds before next poll
    await new Promise(r => setTimeout(r, 2000));
  }
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
