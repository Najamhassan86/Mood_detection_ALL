# Emotion Analytics Dashboard

A production-ready Next.js 14 dashboard for real-time emotion detection analytics from Jetson ONNX systems.

## Features

- **Real-time Emotion Tracking**: Monitor happy, sad, and angry emotions
- **Live Stream Panel**: Placeholder for MJPEG video stream integration
- **Person Tracking**: Individual emotion statistics per person
- **Dark Theme**: Modern, glossy UI with Tailwind CSS
- **TypeScript**: Full type safety throughout the application

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Project Structure

```
mood-dashboard/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main dashboard page
│   └── globals.css         # Global styles
├── types/
│   └── dashboard.ts        # TypeScript type definitions
├── package.json
├── next.config.js
├── tailwind.config.ts
├── tsconfig.json
└── README.md
```

## Data Structure

The dashboard uses the following TypeScript interfaces:

```typescript
type EmotionTotals = { [emotion: string]: number };

type PersonState = {
  person_id: string;
  current_emotion: string;
  time_happy: number;
  time_sad: number;
  time_angry: number;
  last_seen: string;
};

type DashboardSummary = {
  device_id: string;
  device_name: string;
  updated_at: string;
  emotion_totals: EmotionTotals;
  current_people: PersonState[];
};
```

## Current Status

**v0.1** - Mock data implementation

- ✅ Dashboard UI complete
- ✅ Mock data integration
- ⏳ Backend API integration (pending)
- ⏳ Live video stream (pending)

## Future Integration

### Live Stream

To integrate the MJPEG stream from your Jetson device, uncomment and update the image source in the Live Stream Panel:

```jsx
<img src="http://JETSON_IP:5000/video" alt="Live stream" />
```

### API Integration

Replace the `MOCK_DATA` constant with actual API calls to your backend:

```typescript
useEffect(() => {
  const fetchData = async () => {
    const response = await fetch('http://YOUR_BACKEND_API/summary');
    const data = await response.json();
    setData(data);
  };

  fetchData();
  const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds

  return () => clearInterval(interval);
}, []);
```

## Model Information

- **ONNX Model**: enet_b0_8_best_vgaf
- **Device**: Jetson NX

## License

Private project
