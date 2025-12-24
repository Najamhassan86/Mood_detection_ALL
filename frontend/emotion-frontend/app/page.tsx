"use client";

import { useEffect, useState } from "react";

type EmotionTotals = {
  happy: number;
  sad: number;
  angry: number;
  neutral: number;
  [key: string]: number;
};

type PersonState = {
  person_id: string;
  current_emotion: string;
  time_happy: number;
  time_sad: number;
  time_angry: number;
  time_neutral: number;
  last_seen: string;
};

type DashboardSummary = {
  device_id: string;
  device_name: string;
  updated_at: string;
  emotion_totals: EmotionTotals;
  current_people: PersonState[];
  model_fps?: number;
  frontend_fps?: number;
};

// Temporary mock data until backend is reachable
const MOCK_DATA: DashboardSummary = {
  device_id: "jetson_1",
  device_name: "Entrance Camera",
  updated_at: new Date().toISOString(),
  emotion_totals: {
    happy: 1234.5,
    sad: 345.0,
    angry: 120.0,
    neutral: 800.0,
  },
  current_people: [
    {
      person_id: "p1",
      current_emotion: "happy",
      time_happy: 300.5,
      time_sad: 20.0,
      time_angry: 5.0,
      time_neutral: 50.0,
      last_seen: new Date().toISOString(),
    },
    {
      person_id: "p2",
      current_emotion: "sad",
      time_happy: 10.0,
      time_sad: 180.0,
      time_angry: 0.0,
      time_neutral: 30.0,
      last_seen: new Date().toISOString(),
    },
  ],
  model_fps: 30.0,
  frontend_fps: 60.0,
};

// Helper function to get emotion color classes
const getEmotionColors = (emotion: string) => {
  switch (emotion.toLowerCase()) {
    case "happy":
      return "bg-gradient-happy text-yellow-900 shadow-glow-pink";
    case "sad":
      return "bg-gradient-sad text-blue-100 shadow-glow-blue";
    case "angry":
      return "bg-gradient-angry text-red-100 shadow-glow-pink";
    case "neutral":
      return "bg-gradient-neutral text-purple-100 shadow-glow";
    default:
      return "bg-gradient-purple text-white shadow-glow";
  }
};

export default function Home() {
  const BACKEND =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

  const [data, setData] = useState<DashboardSummary | null>(null);
  const [frontendFps, setFrontendFps] = useState<number>(0);

  // Frontend FPS calculation
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();

    const calculateFps = () => {
      frameCount++;
      const currentTime = performance.now();
      const elapsed = (currentTime - lastTime) / 1000; // Convert to seconds

      if (elapsed >= 1.0) {
        const fps = frameCount / elapsed;
        setFrontendFps(fps);
        frameCount = 0;
        lastTime = currentTime;
      }

      requestAnimationFrame(calculateFps);
    };

    const animationId = requestAnimationFrame(calculateFps);
    return () => cancelAnimationFrame(animationId);
  }, []);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(
          `${BACKEND}/api/dashboard/summary?device_id=jetson_1`
        );
        if (!res.ok) {
          console.error("Failed to fetch summary, using mock data. Status:", res.status);
          setData(MOCK_DATA);
          return;
        }
        const json = (await res.json()) as DashboardSummary;
        setData(json);
      } catch (err) {
        console.error("Error fetching dashboard summary, using mock data:", err);
        setData(MOCK_DATA);
      }
    }

    // initial load
    load();

    // auto-refresh every 5 seconds
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  if (!data) {
    return (
      <main className="min-h-screen flex items-center justify-center relative overflow-hidden">
        {/* Animated background orbs */}
        <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-float-reverse" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-blue-500/20 rounded-full blur-3xl animate-pulse-glow" />

        <div className="relative z-10 text-center">
          <div className="inline-block animate-spin-slow">
            <div className="w-16 h-16 border-4 border-purple-500/30 border-t-purple-500 rounded-full" />
          </div>
          <p className="mt-4 text-xl font-semibold gradient-text animate-pulse">Loading dashboard...</p>
        </div>
      </main>
    );
  }

  const { emotion_totals, current_people } = data;

  const totalHappy = emotion_totals["happy"] ?? 0;
  const totalSad = emotion_totals["sad"] ?? 0;
  const totalAngry = emotion_totals["angry"] ?? 0;

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Animated floating background orbs */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl animate-float" />
        <div className="absolute top-40 right-32 w-80 h-80 bg-pink-500/15 rounded-full blur-3xl animate-float-reverse" />
        <div className="absolute bottom-32 left-40 w-96 h-96 bg-blue-500/15 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-20 right-20 w-64 h-64 bg-cyan-500/20 rounded-full blur-3xl animate-pulse-glow" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 py-8 space-y-8">
        {/* Header */}
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between animate-fade-in">
          <div className="space-y-2">
            <h1 className="text-4xl sm:text-5xl font-bold gradient-text tracking-tight">
              Emotion Analytics Dashboard
            </h1>
            <div className="flex items-center gap-2 text-sm text-white/70">
              <div className="glass-effect px-3 py-1.5 rounded-full inline-flex items-center gap-2">
                <div className="w-2 h-2 bg-gradient-purple rounded-full animate-pulse-glow" />
                <span className="font-semibold text-white">{data.device_name}</span>
                <span className="text-white/50">({data.device_id})</span>
              </div>
              <span className="text-white/40">•</span>
              <span className="text-white/60">
                {new Date(data.updated_at).toLocaleString()}
              </span>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="glass-effect px-4 py-2 rounded-xl text-xs text-white/70 font-mono">
              <div className="text-white/90 font-semibold mb-1">v0.1</div>
              <div>ONNX: <span className="text-purple-300">enet_b0_8_best_vgaf</span></div>
            </div>
            <div className="glass-effect px-4 py-2 rounded-xl text-xs font-mono space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-white/60">Model FPS:</span>
                <span className="text-green-400 font-bold">{(data.model_fps || 0).toFixed(1)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-white/60">Frontend FPS:</span>
                <span className="text-cyan-400 font-bold">{frontendFps.toFixed(1)}</span>
              </div>
            </div>
          </div>
        </header>

        {/* Statistics Cards */}
        <section className="grid gap-6 md:grid-cols-3">
          {/* Happy Card */}
          <div
            className="group relative glass-effect-strong rounded-2xl p-6 overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-glow-lg animate-slide-in-up cursor-pointer"
            style={{ animationDelay: '0.1s' }}
          >
            <div className="absolute inset-0 bg-gradient-happy opacity-10 group-hover:opacity-20 transition-opacity duration-300" />
            <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-400/10 rounded-full blur-2xl group-hover:bg-yellow-400/20 transition-all duration-300 animate-pulse-glow" />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs uppercase tracking-wider text-yellow-200/80 font-semibold">
                  Total Happy Time
                </p>
                <div className="text-2xl">😊</div>
              </div>
              <p className="text-4xl font-bold bg-gradient-happy bg-clip-text text-transparent">
                {totalHappy.toFixed(1)}s
              </p>
              <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-happy rounded-full animate-pulse" style={{ width: '75%' }} />
              </div>
            </div>
          </div>

          {/* Sad Card */}
          <div
            className="group relative glass-effect-strong rounded-2xl p-6 overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-glow-blue animate-slide-in-up cursor-pointer"
            style={{ animationDelay: '0.2s' }}
          >
            <div className="absolute inset-0 bg-gradient-sad opacity-10 group-hover:opacity-20 transition-opacity duration-300" />
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-400/10 rounded-full blur-2xl group-hover:bg-blue-400/20 transition-all duration-300 animate-pulse-glow" />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs uppercase tracking-wider text-blue-200/80 font-semibold">
                  Total Sad Time
                </p>
                <div className="text-2xl">😢</div>
              </div>
              <p className="text-4xl font-bold bg-gradient-sad bg-clip-text text-transparent">
                {totalSad.toFixed(1)}s
              </p>
              <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-sad rounded-full animate-pulse" style={{ width: '45%' }} />
              </div>
            </div>
          </div>

          {/* Angry Card */}
          <div
            className="group relative glass-effect-strong rounded-2xl p-6 overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-glow-pink animate-slide-in-up cursor-pointer"
            style={{ animationDelay: '0.3s' }}
          >
            <div className="absolute inset-0 bg-gradient-angry opacity-10 group-hover:opacity-20 transition-opacity duration-300" />
            <div className="absolute top-0 right-0 w-32 h-32 bg-red-400/10 rounded-full blur-2xl group-hover:bg-red-400/20 transition-all duration-300 animate-pulse-glow" />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs uppercase tracking-wider text-red-200/80 font-semibold">
                  Total Angry Time
                </p>
                <div className="text-2xl">😠</div>
              </div>
              <p className="text-4xl font-bold bg-gradient-angry bg-clip-text text-transparent">
                {totalAngry.toFixed(1)}s
              </p>
              <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-angry rounded-full animate-pulse" style={{ width: '30%' }} />
              </div>
            </div>
          </div>
        </section>

        {/* Main Content Grid */}
        <section className="grid gap-6 lg:grid-cols-2">
          {/* Live Stream Panel */}
          <div
            className="glass-effect-strong rounded-2xl p-6 flex flex-col gap-4 animate-slide-in-up transition-all duration-300 hover:shadow-glow"
            style={{ animationDelay: '0.4s' }}
          >
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <span className="w-1 h-8 bg-gradient-purple rounded-full" />
                Live Stream
              </h2>
              <div className="relative">
                <div className="absolute inset-0 bg-emerald-500/30 rounded-full blur-md animate-pulse-glow" />
                <span className="relative inline-flex items-center gap-2 glass-effect px-4 py-2 rounded-full text-sm text-emerald-300 font-semibold">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500" />
                  </span>
                  Live
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-sm text-white/60 flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                MJPEG stream proxied through backend
              </p>
              <div className="glass-effect px-3 py-1.5 rounded-lg text-xs font-mono">
                <span className="text-white/60">Stream FPS: </span>
                <span className="text-green-400 font-bold">{(data.model_fps || 0).toFixed(1)}</span>
              </div>
            </div>
            <div className="relative mt-2 aspect-video w-full overflow-hidden rounded-xl glass-effect group cursor-pointer">
              <div className="absolute inset-0 bg-gradient-purple opacity-0 group-hover:opacity-10 transition-opacity duration-300 z-10" />
              <div className="absolute inset-0 border-2 border-purple-500/0 group-hover:border-purple-500/50 rounded-xl transition-all duration-300 z-10" />
              <img
                src={`${BACKEND}/stream/jetson_1`}
                alt="Live emotion stream"
                className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
              />
            </div>
          </div>

          {/* Current People Table */}
          <div
            className="glass-effect-strong rounded-2xl p-6 animate-slide-in-up transition-all duration-300 hover:shadow-glow"
            style={{ animationDelay: '0.5s' }}
          >
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <span className="w-1 h-8 bg-gradient-pink rounded-full" />
              Current People
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="border-b border-white/10">
                  <tr>
                    <th className="py-3 pr-4 text-left text-xs uppercase tracking-wider text-white/60 font-semibold">ID</th>
                    <th className="py-3 pr-4 text-left text-xs uppercase tracking-wider text-white/60 font-semibold">Emotion</th>
                    <th className="py-3 pr-4 text-right text-xs uppercase tracking-wider text-white/60 font-semibold">Happy</th>
                    <th className="py-3 pr-4 text-right text-xs uppercase tracking-wider text-white/60 font-semibold">Sad</th>
                    <th className="py-3 pr-4 text-right text-xs uppercase tracking-wider text-white/60 font-semibold">Angry</th>
                    <th className="py-3 pr-4 text-right text-xs uppercase tracking-wider text-white/60 font-semibold">Neutral</th>
                    <th className="py-3 pr-4 text-left text-xs uppercase tracking-wider text-white/60 font-semibold">Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {current_people.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-8 text-center">
                        <div className="flex flex-col items-center gap-3">
                          <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center">
                            <span className="text-3xl opacity-50">👥</span>
                          </div>
                          <p className="text-white/40">No people currently detected</p>
                        </div>
                      </td>
                    </tr>
                  ) : (
                    current_people.map((p, index) => (
                      <tr
                        key={p.person_id}
                        className="border-b border-white/5 last:border-b-0 group hover:bg-white/5 transition-all duration-200 animate-fade-in"
                        style={{ animationDelay: `${0.1 * index}s` }}
                      >
                        <td className="py-3 pr-4">
                          <span className="font-mono text-xs glass-effect px-2 py-1 rounded-md">
                            {p.person_id}
                          </span>
                        </td>
                        <td className="py-3 pr-4">
                          <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold capitalize transition-all duration-300 group-hover:scale-110 ${getEmotionColors(p.current_emotion)}`}>
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                            {p.current_emotion}
                          </span>
                        </td>
                        <td className="py-3 pr-4 text-right font-medium text-white/80">
                          {p.time_happy.toFixed(1)}s
                        </td>
                        <td className="py-3 pr-4 text-right font-medium text-white/80">
                          {p.time_sad.toFixed(1)}s
                        </td>
                        <td className="py-3 pr-4 text-right font-medium text-white/80">
                          {p.time_angry.toFixed(1)}s
                        </td>
                        <td className="py-3 pr-4 text-right font-medium text-white/80">
                          {p.time_neutral.toFixed(1)}s
                        </td>
                        <td className="py-3 pr-4 text-xs text-white/50 font-mono">
                          {new Date(p.last_seen).toLocaleTimeString()}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
