"use client";

import { motion } from "framer-motion";

interface Props {
  result: {
    score: number;
    label: "low" | "medium" | "high";
    explanation: string;
  };
}

const RISK_CONFIG = {
  low: { icon: "🟢", title: "Low Hallucination Risk" },
  medium: { icon: "🟡", title: "Moderate Hallucination Risk" },
  high: { icon: "🔴", title: "High Hallucination Risk" },
};

export default function RiskBanner({ result }: Props) {
  const cfg = RISK_CONFIG[result.label];

  return (
    <motion.div
      className={`risk-banner ${result.label}`}
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      <span className="risk-icon">{cfg.icon}</span>
      <div className="risk-info">
        <h3>
          {cfg.title} — <code style={{ fontSize: "1rem" }}>{result.score.toFixed(3)}</code>
        </h3>
        <p>{result.explanation}</p>
      </div>
    </motion.div>
  );
}
