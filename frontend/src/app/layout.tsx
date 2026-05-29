import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Lie Detector — Hallucination Detection Dashboard",
  description:
    "Detects hallucinations in LLM responses using calibration scoring, semantic uncertainty analysis, and multi-model NLI cross-checking.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
