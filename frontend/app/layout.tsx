import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OptiServe | Deep Learning Reasoning Distillation & Speculative Decoding Engine",
  description:
    "Production-grade PyTorch inference acceleration platform featuring exact rejection-sampling speculative decoding, 5-way quantization benchmarking (AWQ, GPTQ, GGUF, FP8, FP16), and AirLLM layer-wise NVMe streaming on RTX 4060 GPU.",
  keywords: [
    "Machine Learning Systems",
    "Inference Engine",
    "Speculative Decoding",
    "PyTorch",
    "Quantization",
    "AirLLM",
    "RTX 4060",
    "FastAPI",
    "Next.js",
  ],
  authors: [{ name: "Hamza Hattab", url: "https://github.com/hamzahattab" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-dark-950 font-sans text-slate-100 antialiased selection:bg-brand-emerald/30 selection:text-brand-emerald">
        {children}
      </body>
    </html>
  );
}
