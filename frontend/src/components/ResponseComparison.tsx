"use client";

interface Props {
  data: {
    local_response: string;
    groq_response: string | null;
    groq_available: boolean;
    groq_model: string;
    error: string | null;
    error_type?: string;
    verdict: string;
    symmetric_agreement: number;
    ab_detail: Record<string, number>;
    ba_detail: Record<string, number>;
  };
}

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function ResponseComparison({ data }: Props) {
  const verdictClass = data.groq_available
    ? data.verdict
    : "unavailable";

  const nliData =
    data.ab_detail && data.ba_detail && Object.keys(data.ab_detail).length > 0
      ? [
          {
            name: "Contradiction",
            "Local → Groq": data.ab_detail.contradiction || 0,
            "Groq → Local": data.ba_detail.contradiction || 0,
          },
          {
            name: "Entailment",
            "Local → Groq": data.ab_detail.entailment || 0,
            "Groq → Local": data.ba_detail.entailment || 0,
          },
          {
            name: "Neutral",
            "Local → Groq": data.ab_detail.neutral || 0,
            "Groq → Local": data.ba_detail.neutral || 0,
          },
        ]
      : null;

  return (
    <div>
      <div className="response-grid">
        {/* Local Model */}
        <div className="response-card">
          <h4>
            <span style={{ color: "var(--accent-purple)" }}>●</span> Local Model
            (TinyLlama)
          </h4>
          <p>{data.local_response || "No response generated"}</p>
        </div>

        {/* Groq Model */}
        <div className="response-card">
          <h4>
            <span style={{ color: "var(--accent-emerald)" }}>●</span>{" "}
            {data.groq_model || "Groq"}
          </h4>
          {data.groq_available ? (
            <p>{data.groq_response || "—"}</p>
          ) : (
            <div>
              <p style={{ color: "var(--accent-amber)", marginBottom: 8 }}>
                ⚠️ Groq cross-check unavailable
              </p>
              {data.error_type === "invalid_key" ||
              (data.error && data.error.includes("401")) ? (
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                  🔑 Invalid Groq API Key. Go to{" "}
                  <a
                    href="https://console.groq.com/keys"
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: "var(--accent-indigo-light)" }}
                  >
                    console.groq.com/keys
                  </a>{" "}
                  to create a new key.
                </p>
              ) : (
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                  Error: {data.error}
                </p>
              )}
              <p
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-muted)",
                  marginTop: 8,
                }}
              >
                Running in 2-signal mode (calibration + uncertainty only)
              </p>
            </div>
          )}
        </div>
      </div>

      {/* NLI Verdict */}
      {data.groq_available && (
        <div style={{ marginTop: 12 }}>
          <span className={`nli-verdict ${verdictClass}`}>
            {data.verdict === "agree" && "🟢"}
            {data.verdict === "neutral" && "🟡"}
            {data.verdict === "contradict" && "🔴"}
            {" "}{data.verdict.toUpperCase()}
          </span>
          <span
            style={{
              marginLeft: 16,
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
            }}
          >
            Symmetric agreement:{" "}
            <code>{data.symmetric_agreement.toFixed(3)}</code>
          </span>
        </div>
      )}

      {/* NLI Scores Chart */}
      {nliData && (
        <div className="chart-container" style={{ marginTop: 16 }}>
          <h4>Bidirectional NLI Probabilities</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={nliData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              />
              <Tooltip
                contentStyle={{
                  background: "#1a1d2b",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  color: "#e8eaf0",
                  fontSize: 13,
                }}
                formatter={(value: any) => (value || 0).toFixed(4)}
              />
              <Legend
                wrapperStyle={{ fontSize: 12, color: "#9ca3af" }}
              />
              <Bar dataKey="Local → Groq" fill="#a855f7" radius={[4, 4, 0, 0]} barSize={30} />
              <Bar dataKey="Groq → Local" fill="#10b981" radius={[4, 4, 0, 0]} barSize={30} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
