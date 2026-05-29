"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Detector", icon: "🔍" },
  { href: "/analytics", label: "Analytics", icon: "📊" },
];

export default function Sidebar({ activePage }: { activePage?: string }) {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span style={{ fontSize: "1.4rem" }}>🛡️</span>
        <h1>LLM Lie Detector</h1>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const isActive =
            activePage === item.label.toLowerCase() || pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-link ${isActive ? "active" : ""}`}
            >
              <span>{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div style={{ flex: 1 }} />

      <div
        style={{
          padding: "12px 16px",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
          borderTop: "1px solid var(--border-subtle)",
          marginTop: 16,
        }}
      >
        <p>3-signal hallucination detection</p>
        <p style={{ marginTop: 4, opacity: 0.6 }}>
          Calibration · Semantic · NLI
        </p>
      </div>
    </aside>
  );
}
