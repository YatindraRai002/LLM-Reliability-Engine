"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  data?: {
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
}

const SIGNAL_COLORS: Record<string, string> = {
  calibration: "#a855f7",
  semantic_uncertainty: "#f43f5e",
  cross_check: "#10b981",
};

function getTokenClass(severity: string): string {
  switch (severity) {
    case "critical":
      return "token-highlight token-critical";
    case "warning":
      return "token-highlight token-warning";
    case "ok":
      return "token-highlight token-ok";
    default:
      return "token-highlight token-neutral";
  }
}

export default function ExplanationsPanel({ data }: Props) {
  if (!data || typeof data !== "object") {
    return (
      <div className="empty-state">
        <div className="icon">🔍</div>
        <h3>Explanation data is not available</h3>
        <p>
          This happens when the explainer module didn&apos;t run. Check that
          explain_result() is called in the backend pipeline.
        </p>
      </div>
    );
  }

  const hasContent =
    data.flagged_spans?.length > 0 ||
    (data.signal_pct && Object.keys(data.signal_pct).length > 0) ||
    data.recommendations?.length > 0 ||
    data.highlighted_html;

  if (!hasContent) {
    return (
      <div className="empty-state">
        <div className="icon">✅</div>
        <h3>No significant explanation signals</h3>
        <p>The response appears clean across all detection signals.</p>
      </div>
    );
  }

  // Signal contribution data
  const signalData = data.signal_pct
    ? Object.entries(data.signal_pct).map(([name, value]) => ({
        name: name
          .replace("_", " ")
          .replace(/\b\w/g, (l) => l.toUpperCase()),
        value: value as number,
        key: name,
      }))
    : [];

  return (
    <div>
      {/* 1. Token-Level Confidence HTML */}
      {data.highlighted_html && (
        <div className="chart-container">
          <h4>🔍 Token-Level Confidence</h4>
          <div
            dangerouslySetInnerHTML={{ __html: data.highlighted_html }}
            style={{ lineHeight: 1.8, padding: 8 }}
          />
          <p
            style={{
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              marginTop: 8,
            }}
          >
            Each token colored by model confidence. Red = uncertain, Green =
            confident.
          </p>
        </div>
      )}

      {/* 2. Signal Contribution */}
      {signalData.length > 0 && (
        <div className="chart-container" style={{ marginTop: 16 }}>
          <h4>📊 Signal Contribution Analysis</h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={signalData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.06)"
              />
              <XAxis
                type="number"
                domain={[0, 100]}
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                unit="%"
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                width={150}
              />
              <Tooltip
                contentStyle={{
                  background: "#1a1d2b",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  color: "#e8eaf0",
                  fontSize: 13,
                }}
                formatter={(value: any) => [`${(value || 0).toFixed(1)}%`, "Contribution"]}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
                {signalData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={SIGNAL_COLORS[entry.key] || "#6366f1"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p
            style={{
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              marginTop: 4,
            }}
          >
            Which signal drove the hallucination score?
          </p>
        </div>
      )}

      {/* 3. Contradicting Sentences */}
      <div className="chart-container" style={{ marginTop: 16 }}>
        <h4>🔴 Contradicting Sentences</h4>
        {data.contradicting_sentences && data.contradicting_sentences.length > 0 ? (
          data.contradicting_sentences.map((cs, i) => (
            <div key={i} className="contradiction-card">
              <div className="score">
                Contradiction Score: {cs.contradiction_score.toFixed(3)}
              </div>
              <div className="text">{cs.sentence}</div>
            </div>
          ))
        ) : (
          <p
            style={{
              color: "var(--accent-emerald)",
              fontSize: "0.9rem",
              padding: "8px 0",
            }}
          >
            ✅ No contradicting sentences detected.
          </p>
        )}
      </div>

      {/* 4. Recommendations */}
      <div className="chart-container" style={{ marginTop: 16 }}>
        <h4>💡 Recommendations</h4>
        {data.recommendations && data.recommendations.length > 0 ? (
          data.recommendations.map((rec, i) => (
            <div key={i} className="recommendation">
              {rec}
            </div>
          ))
        ) : (
          <p
            style={{
              color: "var(--accent-emerald)",
              fontSize: "0.9rem",
              padding: "8px 0",
            }}
          >
            ✅ No actionable recommendations.
          </p>
        )}
      </div>
    </div>
  );
}
