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
  result: {
    calibration_score: number;
    uncertainty_score: number;
    cross_check_score: number;
  };
}

const COLORS = ["#a855f7", "#f43f5e", "#10b981"];

export default function SignalBreakdown({ result }: Props) {
  const data = [
    { name: "Calibration", value: result.calibration_score },
    { name: "Semantic Uncertainty", value: result.uncertainty_score },
    { name: "Cross-Check", value: result.cross_check_score },
  ];

  return (
    <div className="chart-container">
      <h4>Signal Breakdown</h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
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
            formatter={(value: any) => [(value || 0).toFixed(3), "Score"]}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={50}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
