"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Sidebar from "@/components/Sidebar";
import RiskBanner from "@/components/RiskBanner";
import MetricCards from "@/components/MetricCards";
import SignalBreakdown from "@/components/SignalBreakdown";
import ResponseComparison from "@/components/ResponseComparison";
import UncertaintyLandscape from "@/components/UncertaintyLandscape";
import TokenConfidence from "@/components/TokenConfidence";
import ExplanationsPanel from "@/components/ExplanationsPanel";
import { analyzeQuery, type AnalyzeResult } from "@/lib/api";

const TABS = [
  { id: "responses", label: "Model Responses", icon: "💬" },
  { id: "uncertainty", label: "Uncertainty Landscape", icon: "🌐" },
  { id: "tokens", label: "Token Confidence", icon: "📊" },
  { id: "explanations", label: "Explanations", icon: "🔍" },
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("responses");

  const handleAnalyze = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeQuery(query.trim());
      setResult(data);
      setActiveTab("responses");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const renderTabContent = () => {
    if (!result) return null;
    switch (activeTab) {
      case "responses":
        return <ResponseComparison data={result.cross_check_detail} />;
      case "uncertainty":
        return <UncertaintyLandscape data={result.uncertainty_detail} />;
      case "tokens":
        return <TokenConfidence data={result.calibration_detail} />;
      case "explanations":
        return <ExplanationsPanel data={result.explanation_detail} />;
      default:
        return null;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar activePage="detector" />
      <main className="main-content">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h2 style={{ fontSize: "1.6rem", fontWeight: 700, marginBottom: 4 }}>
            LLM Lie Detector
          </h2>
          <p
            style={{
              color: "var(--text-muted)",
              fontSize: "0.9rem",
              marginBottom: 28,
            }}
          >
            Detect hallucinations using calibration scoring, semantic
            uncertainty, and multi-model cross-checking.
          </p>

          {/* Query Input */}
          <div className="query-input-wrapper">
            <input
              className="query-input"
              type="text"
              placeholder="e.g. Who invented the telephone?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              disabled={loading}
              maxLength={10000}
            />
            <button
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={loading || !query.trim()}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{
                marginTop: 16,
                padding: "12px 20px",
                background: "rgba(244,63,94,0.1)",
                border: "1px solid rgba(244,63,94,0.3)",
                borderRadius: "var(--radius-md)",
                color: "#fb7185",
                fontSize: "0.9rem",
              }}
            >
              {error}
            </motion.div>
          )}

          {/* Loading Spinner */}
          {loading && (
            <div className="spinner-overlay">
              <div className="spinner" />
              <p className="spinner-text">
                Running 3-signal analysis pipeline...
              </p>
            </div>
          )}

          {/* Results */}
          <AnimatePresence>
            {result && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <RiskBanner result={result.result} />
                <MetricCards result={result.result} />
                <SignalBreakdown result={result.result} />

                {/* Tabs */}
                <div className="tabs-container">
                  <div className="tabs-header">
                    {TABS.map((tab) => (
                      <button
                        key={tab.id}
                        className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
                        onClick={() => setActiveTab(tab.id)}
                      >
                        {tab.icon} {tab.label}
                      </button>
                    ))}
                  </div>
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeTab}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      {renderTabContent()}
                    </motion.div>
                  </AnimatePresence>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>
    </div>
  );
}
