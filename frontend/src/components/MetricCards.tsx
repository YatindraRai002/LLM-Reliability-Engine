"use client";

import { motion } from "framer-motion";

interface Props {
  result: {
    score: number;
    calibration_score: number;
    uncertainty_score: number;
    cross_check_score: number;
  };
}

export default function MetricCards({ result }: Props) {
  const metrics = [
    { label: "Final Score", value: result.score, color: "var(--accent-indigo-light)" },
    { label: "Calibration", value: result.calibration_score, color: "var(--accent-purple)" },
    { label: "Uncertainty", value: result.uncertainty_score, color: "var(--accent-rose)" },
    { label: "Cross-Check", value: result.cross_check_score, color: "var(--accent-emerald)" },
  ];

  return (
    <div className="metrics-row">
      {metrics.map((m, i) => (
        <motion.div
          key={m.label}
          className="metric-card"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 + i * 0.08 }}
        >
          <div className="label">{m.label}</div>
          <div className="value" style={{ color: m.color }}>
            {m.value.toFixed(3)}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
