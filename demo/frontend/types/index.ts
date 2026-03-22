export type JournalEntryData = {
  journal_text: string;
  ambience_type: string;
  time_of_day: string;
  duration_min: number;
  sleep_hours: number | null;
  energy_level: number;
  stress_level: number;
  previous_day_mood: string | null;
  face_emotion_hint: string | null;
  reflection_quality: string;
};

export type BackendResponse = {
  predicted_state: string;
  predicted_intensity: number;
  confidence: number;
  uncertain_flag: boolean;
  what_to_do: string;
  when_to_do: string;
};
