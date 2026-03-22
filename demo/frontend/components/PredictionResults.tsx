"use client";

import { BackendResponse } from "../types";

export default function PredictionResults({ result }: { result: BackendResponse | null }) {
  return (
    <section className="lg:col-span-4 h-full">
      <div className={`p-8 rounded-2xl border transition-all duration-500 h-full ${result ? 'bg-linear-to-b from-indigo-50/50 to-white dark:from-indigo-950/20 dark:to-neutral-900 border-indigo-100 dark:border-indigo-900 shadow-xl shadow-indigo-500/5' : 'bg-transparent border-dashed border-neutral-300 dark:border-neutral-800 flex flex-col items-center justify-center text-center'}`}>
        {!result ? (
          <div className="text-neutral-400 dark:text-neutral-500">
            <div className="mb-4 opacity-50">
              <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
            </div>
            <p className="text-sm font-medium">Results will appear here</p>
            <p className="text-xs mt-1">Submit the form to see predictions</p>
          </div>
        ) : (
          <div className="space-y-6 animate-in slide-in-from-bottom-4 fade-in duration-500">
            <div className="flex items-center justify-between border-b border-indigo-100 dark:border-neutral-800 pb-4">
              <h3 className="text-lg font-bold text-neutral-900 dark:text-white">Predictions</h3>
              <span className="inline-flex items-center rounded-full bg-green-50 px-2.5 py-1 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-600/20 dark:bg-green-900/30 dark:text-green-400 dark:ring-green-500/20">
                Success
              </span>
            </div>

            <div className="space-y-5">
              <div>
                <p className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-1">Emotional State</p>
                <div className="flex items-end gap-2">
                  <p className="text-2xl font-black tracking-tight text-indigo-600 dark:text-indigo-400 capitalize">{result.predicted_state}</p>
                  <p className="text-sm font-medium text-neutral-400 mb-1">{(result.confidence * 100).toFixed(0)}% conf.</p>
                </div>
                <div className="w-full bg-neutral-200 dark:bg-neutral-800 rounded-full h-1.5 mt-2">
                  <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${result.confidence * 100}%` }}></div>
                </div>
              </div>

              <div>
                <p className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-1">Intensity</p>
                <div className="flex items-end gap-2">
                  <p className="text-2xl font-black tracking-tight text-cyan-600 dark:text-cyan-400">{result.predicted_intensity} / 5</p>
                </div>
              </div>

              <div>
                <p className="text-sm font-medium text-neutral-500 dark:text-neutral-400 mb-1">Recommended Action</p>
                <div className="p-3 bg-neutral-100 dark:bg-neutral-800 rounded-xl">
                  <p className="text-md font-semibold text-neutral-900 dark:text-white capitalize">{result.what_to_do.replace(/_/g, " ")}</p>
                  <p className="text-sm text-neutral-500 dark:text-neutral-400 mt-1 capitalize">Timing: {result.when_to_do.replace(/_/g, " ")}</p>
                </div>
              </div>

              <div className="pt-2">
                <div className={`rounded-xl p-4 border ${
                  result.uncertain_flag
                    ? 'bg-amber-50 border-amber-200 dark:bg-amber-950/30 dark:border-amber-900'
                    : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900'
                }`}>
                  <div className="flex items-start">
                    <div className="shrink-0 mt-0.5">
                      {result.uncertain_flag ? (
                        <svg className="h-5 w-5 text-amber-600 dark:text-amber-500" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      ) : (
                        <svg className="h-5 w-5 text-emerald-600 dark:text-emerald-500" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                    <div className="ml-3">
                      <h3 className={`text-sm font-semibold ${
                        result.uncertain_flag
                          ? 'text-amber-800 dark:text-amber-300'
                          : 'text-emerald-800 dark:text-emerald-300'
                      }`}>
                        Uncertainty Flag
                      </h3>
                      <div className={`mt-1 text-sm ${
                        result.uncertain_flag
                          ? 'text-amber-700 dark:text-amber-400'
                          : 'text-emerald-700 dark:text-emerald-400'
                      }`}>
                        <p>
                          {result.uncertain_flag
                            ? 'High uncertainty detected in predictions. Model requires human review.'
                            : 'Low uncertainty detected. Predictions are deemed stable.'
                          }
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
