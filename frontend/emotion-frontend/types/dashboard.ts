export type EmotionTotals = { [emotion: string]: number };

export type PersonState = {
  person_id: string;
  current_emotion: string;
  time_happy: number;
  time_sad: number;
  time_angry: number;
  last_seen: string;
};

export type DashboardSummary = {
  device_id: string;
  device_name: string;
  updated_at: string;
  emotion_totals: EmotionTotals;
  current_people: PersonState[];
};
