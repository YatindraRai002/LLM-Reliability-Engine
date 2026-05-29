"use client";

import { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  data: {
    uncertainty_score: number;
    n_semantic_clusters: number;
    mean_pairwise_similarity: number;
    cluster_labels: number[];
    embeddings_2d: number[][];
    responses: string[];
  };
}

const CLUSTER_COLORS = [
  "#a855f7", "#f43f5e", "#10b981", "#f59e0b", "#06b6d4", "#6366f1",
];

export default function UncertaintyLandscape({ data }: Props) {
  const [expanded, setExpanded] = useState(false);

  const metrics = [
    { label: "Uncertainty Score", value: data.uncertainty_score.toFixed(3) },
    { label: "Semantic Clusters", value: data.n_semantic_clusters },
    { label: "Mean Similarity", value: data.mean_pairwise_similarity.toFixed(3) },
  ];

  // Build scatter data
  const scatterData =
    data.embeddings_2d && data.embeddings_2d.length > 0
      ? data.embeddings_2d.map((coords, i) => ({
          x: coords[0],
          y: coords[1],
          cluster: data.cluster_labels[i] ?? 0,
          label: `R${i}`,
          response: data.responses[i]?.substring(0, 100) || "",
        }))
      : null;

  return (
    <div>
      {/* Metrics */}
      <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
        {metrics.map((m) => (
          <div key={m.label} className="metric-card" style={{ flex: 1 }}>
            <div className="label">{m.label}</div>
            <div className="value" style={{ fontSize: "1.3rem" }}>
              {m.value}
            </div>
          </div>
        ))}
      </div>

      {/* Scatter Plot */}
      {scatterData && scatterData.length > 0 ? (
        <div className="chart-container">
          <h4>
            Response Clusters — {data.n_semantic_clusters} semantic group
            {data.n_semantic_clusters > 1 ? "s" : ""} (uncertainty=
            {data.uncertainty_score.toFixed(3)})
          </h4>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis
                dataKey="x"
                type="number"
                name="PCA dim 1"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                label={{
                  value: "PCA dim 1",
                  position: "bottom",
                  fill: "#6b7280",
                  fontSize: 12,
                }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="PCA dim 2"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                label={{
                  value: "PCA dim 2",
                  angle: -90,
                  position: "left",
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
                  fontSize: 12,
                }}
                formatter={(_: unknown, __: any, props: any) => [
                  props?.payload?.response + "...",
                  "Response",
                ]}
              />
              <Scatter data={scatterData} name="Responses">
                {scatterData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                    r={8}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <p
            style={{
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              marginTop: 8,
            }}
          >
            Each point = one sampled response. Tight cluster = consistent model.
            Scattered = uncertain.
          </p>
        </div>
      ) : (
        <div className="empty-state">
          <div className="icon">🌐</div>
          <h3>No embedding data</h3>
          <p>Fewer than 2 samples were generated for clustering.</p>
        </div>
      )}

      {/* Sampled Responses Expander */}
      {data.responses && data.responses.length > 0 && (
        <div className="expander">
          <div
            className="expander-header"
            onClick={() => setExpanded(!expanded)}
          >
            <span>
              All {data.responses.length} sampled responses
            </span>
            <span>{expanded ? "▲" : "▼"}</span>
          </div>
          {expanded && (
            <div className="expander-body">
              {data.responses.map((resp, i) => (
                <div key={i} className="sample-response">
                  <div className="sample-label">
                    Sample {i + 1} (cluster {data.cluster_labels[i] ?? 0})
                  </div>
                  <div className="sample-text">{resp}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
