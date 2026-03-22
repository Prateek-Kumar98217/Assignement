"use client";

import { JournalEntryData } from "../types";

type JournalFormProps = {
  onSubmit: (data: JournalEntryData) => void;
  loading: boolean;
};

export default function JournalForm({ onSubmit, loading }: JournalFormProps) {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const rawData = Object.fromEntries(formData.entries());

    const data: JournalEntryData = {
      journal_text: rawData.journal_text as string,
      ambience_type: rawData.ambience_type as string,
      time_of_day: rawData.time_of_day as string,
      duration_min: parseInt(rawData.duration_min as string, 10),
      sleep_hours: rawData.sleep_hours ? parseFloat(rawData.sleep_hours as string) : null,
      energy_level: parseInt(rawData.energy_level as string, 10),
      stress_level: parseInt(rawData.stress_level as string, 10),
      previous_day_mood: rawData.previous_day_mood ? (rawData.previous_day_mood as string) : null,
      face_emotion_hint: rawData.face_emotion_hint ? (rawData.face_emotion_hint as string) : null,
      reflection_quality: rawData.reflection_quality as string,
    };

    onSubmit(data);
  };

  return (
    <section className="lg:col-span-8 bg-white dark:bg-neutral-900 p-8 rounded-2xl shadow-sm border border-neutral-200 dark:border-neutral-800 transition-all">
      <h2 className="text-2xl font-bold mb-6">Journal Entry</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="journal_text" className="block text-sm font-medium">
            Journal Text
          </label>
          <textarea
            id="journal_text"
            name="journal_text"
            rows={4}
            required
            className="block w-full rounded-xl border-0 py-3 px-4 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 placeholder:text-neutral-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all resize-y"
            placeholder="How are you feeling today?"
          ></textarea>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 hover:gap-6">
          <div className="space-y-2">
            <label htmlFor="ambience_type" className="block text-sm font-medium">
              Ambience Type
            </label>
            <select
              id="ambience_type"
              name="ambience_type"
              className="block w-full rounded-xl border-0 py-3 pl-4 pr-10 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            >
              <option value="cafe">Cafe</option>
              <option value="forest">Forest</option>
              <option value="mountain">Mountain</option>
              <option value="ocean">Ocean</option>
              <option value="rain">Rain</option>
            </select>
          </div>

          <div className="space-y-2">
            <label htmlFor="time_of_day" className="block text-sm font-medium">
              Time of Day
            </label>
            <select
              id="time_of_day"
              name="time_of_day"
              className="block w-full rounded-xl border-0 py-3 pl-4 pr-10 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            >
              <option value="morning">Morning</option>
              <option value="afternoon">Afternoon</option>
              <option value="evening">Evening</option>
              <option value="night">Night</option>
              <option value="early_morning">Early Morning</option>
            </select>
          </div>

          <div className="space-y-2">
            <label htmlFor="duration_min" className="block text-sm font-medium">
              Duration (mins)
            </label>
            <input
              type="number"
              id="duration_min"
              name="duration_min"
              required
              min="1"
              className="block w-full rounded-xl border-0 py-3 px-4 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
              placeholder="e.g. 15"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="sleep_hours" className="block text-sm font-medium">
              Sleep Hours
            </label>
            <input
              type="number"
              step="0.1"
              id="sleep_hours"
              name="sleep_hours"
              required
              min="0"
              max="24"
              className="block w-full rounded-xl border-0 py-3 px-4 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
              placeholder="e.g. 7.5"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="energy_level" className="block text-sm font-medium">
              Energy Level (1-5)
            </label>
            <input
              type="number"
              id="energy_level"
              name="energy_level"
              required
              min="1"
              max="5"
              className="block w-full rounded-xl border-0 py-3 px-4 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="stress_level" className="block text-sm font-medium">
              Stress Level (1-5)
            </label>
            <input
              type="number"
              id="stress_level"
              name="stress_level"
              required
              min="1"
              max="5"
              className="block w-full rounded-xl border-0 py-3 px-4 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="previous_day_mood" className="block text-sm font-medium">
              Previous Day Mood
            </label>
            <select
              id="previous_day_mood"
              name="previous_day_mood"
              className="block w-full rounded-xl border-0 py-3 pl-4 pr-10 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            >
              <option value="none">None</option>
              <option value="calm">Calm</option>
              <option value="focused">Focused</option>
              <option value="mixed">Mixed</option>
              <option value="neutral">Neutral</option>
              <option value="overwhelmed">Overwhelmed</option>
              <option value="restless">Restless</option>
            </select>
          </div>

          <div className="space-y-2">
            <label htmlFor="face_emotion_hint" className="block text-sm font-medium">
              Face Emotion Hint
            </label>
            <select
              id="face_emotion_hint"
              name="face_emotion_hint"
              className="block w-full rounded-xl border-0 py-3 pl-4 pr-10 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            >
              <option value="none">None</option>
              <option value="calm_face">Calm Face</option>
              <option value="happy_face">Happy Face</option>
              <option value="neutral_face">Neutral Face</option>
              <option value="tense_face">Tense Face</option>
              <option value="tired_face">Tired Face</option>
            </select>
          </div>

          <div className="space-y-2 sm:col-span-2">
            <label htmlFor="reflection_quality" className="block text-sm font-medium">
              Reflection Quality
            </label>
            <select
              id="reflection_quality"
              name="reflection_quality"
              className="block w-full rounded-xl border-0 py-3 pl-4 pr-10 text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-neutral-950 dark:text-white dark:ring-neutral-700 dark:focus:ring-indigo-500 transition-all"
            >
              <option value="clear">Clear</option>
              <option value="conflicted">Conflicted</option>
              <option value="vague">Vague</option>
            </select>
          </div>
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={loading}
            className="w-full flex justify-center py-3.5 px-4 border border-transparent rounded-xl shadow-sm text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-70 disabled:cursor-not-allowed transition-all active:scale-[0.98]"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Predicting...
              </span>
            ) : (
              "Predict State"
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
