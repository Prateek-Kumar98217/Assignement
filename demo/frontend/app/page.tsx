"use client";

import { useState } from "react";
import JournalForm from "../components/JournalForm";
import PredictionResults from "../components/PredictionResults";
import { BackendResponse, JournalEntryData } from "../types";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BackendResponse | null>(null);

  const handleSubmit = async (data: JournalEntryData) => {
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Ensure sleep_hours correctly formats since we parsed it
        // and others as well
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch predictions");
      }

      const json = await response.json() as BackendResponse;
      setResult(json);
    } catch (error) {
      console.error("Error submitting form", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 py-12 px-4 sm:px-6 lg:px-8 font-sans selection:bg-indigo-500/30">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="text-center space-y-2">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl bg-clip-text text-transparent bg-linear-to-r from-indigo-500 to-cyan-500">
            Assignment Demo
          </h1>
          <p className="text-lg text-neutral-500 dark:text-neutral-400 max-w-2xl mx-auto">
            Log your daily metrics to predict emotional state and intensity.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <JournalForm onSubmit={handleSubmit} loading={loading} />
          <PredictionResults result={result} />
        </div>
      </div>
    </main>
  );
}
