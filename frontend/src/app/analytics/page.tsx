"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from "recharts";
import Sidebar from "@/components/Sidebar";
import { getHistory, clearHistory, type HistoryRow } from "@/lib/api";

const LABEL_COLORS: Record<string, string> = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#f43f5e",
};

export default function AnalyticsPage() {
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getHistory();
      setHistory(data.history || []);
    } catch {
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleClear = async () => {
    if (!confirm("Clear all history data?")) return;
    await clearHistory();
    setHistory([]);
  };

  // Computed metrics
  const totalQueries = history.length;
  const highRisk = history.filter((r) => r.label === "high").length;
  const avgScore =
    totalQueries > 0
      ? history.reduce((sum, r) => sum + r.score, 0) / totalQueries
      : 0;
  const avgCal =
    totalQueries > 0
      ? history.reduce((sum, r) => sum + r.cal, 0) / totalQueries
      : 0;

  // Pie chart data
  const labelCounts = { low: 0, medium: 0, high: 0 };
  history.forEach((r) => {
    if (r.label in labelCounts) labelCounts[r.label as keyof typeof labelCounts]++;
  });
  const pieData = Object.entries(labelCounts)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  // Timeline data
  const timelineData = history
    .slice()
    .reverse()
    .map((r) => ({
      time: new Date(r.timestamp).toLocaleDateString(),
      score: r.score,
      label: r.label,
    }));

  // Scatter data
  const scatterData = history.map((r) => ({
    unc: r.unc,
    cc: r.cc,
    label: r.label,
    prompt: r.prompt?.substring(0, 50) || "",
  }));

  return (
    <div className="app-layout">
      <Sidebar activePage="analytics" />
      <main className="main-content">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 28,
            }}
          >
            <div>
              <h2 style={{ fontSize: "1.6rem", fontWeight: 700, marginBottom: 4 }}>
                📊 Usage Analytics
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                Analyze history of scanned prompts, hallucination scores, and
                model performance.
              </p>
            </div>
            {history.length > 0 && (
              <button className="btn-danger" onClick={handleClear}>
                🗑️ Clear History
              </button>
            )}
          </div>

          {loading ? (
            <div className="spinner-overlay">
              <div className="spinner" />
              <p className="spinner-text">Loading analytics...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="empty-state">
              <div className="icon">📊</div>
              <h3>No data available yet</h3>
              <p>Run some queries through the detector to see analytics!</p>
            </div>
          ) : (
            <>
              {/* Metric Cards */}
              <div className="metrics-row">
                <div className="metric-card">
                  <div className="label">Total Queries</div>
                  <div
                    className="value"
                    style={{ color: "var(--accent-indigo-light)" }}
                  >
                    {totalQueries}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="label">High Risk Detected</div>
                  <div className="value" style={{ color: "var(--accent-rose)" }}>
                    {highRisk}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="label">Avg Hallucination Score</div>
                  <div
                    className="value"
                    style={{ color: "var(--accent-amber)" }}
                  >
                    {avgScore.toFixed(2)}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="label">Avg Token Confidence</div>
                  <div
                    className="value"
                    style={{ color: "var(--accent-emerald)" }}
                  >
                    {avgCal.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* Score Timeline */}
              <div className="chart-container">
                <h4>Hallucination Scores Over Time</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={timelineData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.06)"
                    />
                    <XAxis
                      dataKey="time"
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
                      axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fill: "#9ca3af", fontSize: 11 }}
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
                    />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={{ fill: "#6366f1", r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Grid: Pie + Scatter */}
              <div className="analytics-grid">
                <div className="chart-container">
                  <h4>Risk Distribution</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={90}
                        paddingAngle={3}
                        dataKey="value"
                        label={({ name, percent }) =>
                          `${name} ${((percent || 0) * 100).toFixed(0)}%`
                        }
                      >
                        {pieData.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={LABEL_COLORS[entry.name] || "#6366f1"}
                          />
                        ))}
                      </Pie>
                      <Legend
                        wrapperStyle={{ fontSize: 12, color: "#9ca3af" }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-container">
                  <h4>Uncertainty vs Cross-Check</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.06)"
                      />
                      <XAxis
                        dataKey="unc"
                        type="number"
                        name="Uncertainty"
                        domain={[0, 1]}
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      />
                      <YAxis
                        dataKey="cc"
                        type="number"
                        name="Cross-Check"
                        domain={[0, 1]}
                        tick={{ fill: "#9ca3af", fontSize: 11 }}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1a1d2b",
                          border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: 8,
                          color: "#e8eaf0",
                          fontSize: 12,
                        }}
                      />
                      <Scatter data={scatterData} name="Queries">
                        {scatterData.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={LABEL_COLORS[entry.label] || "#6366f1"}
                            r={6}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Recent Queries Table */}
              <div className="chart-container" style={{ marginTop: 20 }}>
                <h4>Recent Queries</h4>
                <div style={{ overflowX: "auto" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Prompt</th>
                        <th>Score</th>
                        <th>Risk</th>
                        <th>Cal</th>
                        <th>Unc</th>
                        <th>CC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.slice(0, 50).map((row) => (
                        <tr key={row.id}>
                          <td>
                            {new Date(row.timestamp).toLocaleString()}
                          </td>
                          <td
                            style={{
                              maxWidth: 300,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {row.prompt}
                          </td>
                          <td>
                            <code>{row.score?.toFixed(3)}</code>
                          </td>
                          <td>
                            <span className={`badge badge-${row.label}`}>
                              {row.label}
                            </span>
                          </td>
                          <td>{row.cal?.toFixed(3)}</td>
                          <td>{row.unc?.toFixed(3)}</td>
                          <td>{row.cc?.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </motion.div>
      </main>
    </div>
  );
}
