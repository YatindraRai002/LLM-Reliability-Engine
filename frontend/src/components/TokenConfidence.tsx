"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  data: {
    response: string;
    token_probs: number[];
    mean_confidence: number;
    n_tokens: number;
    success: boolean;
  };
}

function getTokenColor(p: number): string {
  if (p > 0.7) return "#10b981";
  if (p > 0.4) return "#f59e0b";
  return "#f43f5e";
}

export default function TokenConfidence({ data }: Props) {
  if (!data.token_probs || data.token_probs.length === 0) {
    return (
      <div>
        <div className="empty-state">
          <div className="icon">📊</div>
          <h3>Token probability data not available</h3>
          <p>The model did not produce token-level confidence scores.</p>
        </div>
        {data.response && (
          <div className="chart-container">
            <h4>Generated Response</h4>
            <p style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
              {data.response}
            </p>
          </div>
        )}
      </div>
    );
  }

  const chartData = data.token_probs.map((p, i) => ({
    name: `t${i}`,
    prob: p,
  }));

  return (
    <div>
      <div className="chart-container">
        <h4>
          Token-Level Confidence ({data.n_tokens} tokens, mean={" "}
          {data.mean_confidence.toFixed(3)})
        </h4>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="name"
              tick={{ fill: "#9ca3af", fontSize: 9 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              interval={Math.floor(chartData.length / 20)}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              label={{
                value: "P(token)",
                angle: -90,
                position: "insideLeft",
                fill: "#6b7280",
                fontSize: 12,
              }}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1d2b",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
                color: "#e8eaf0",
                fontSize: 13,
              }}
              formatter={(value: any) => [(value || 0).toFixed(4), "Probability"]}
            />
            <ReferenceLine
              y={0.5}
              stroke="#6b7280"
              strokeDasharray="5 5"
              label={{ value: "50%", fill: "#6b7280", fontSize: 11 }}
            />
            <Bar dataKey="prob" radius={[2, 2, 0, 0]} barSize={6}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={getTokenColor(entry.prob)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 8 }}>
          Green = confident (P &gt; 0.7) · Yellow = moderate · Red = uncertain (P &lt; 0.4)
        </p>
      </div>

      <div className="chart-container" style={{ marginTop: 16 }}>
        <h4>Generated Response</h4>
        <p style={{ color: "var(--text-secondary)", lineHeight: 1.7, fontSize: "0.9rem" }}>
          {data.response}
        </p>
      </div>
    </div>
  );
}
