import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        purple: {
          450: '#9F7AEA',
          550: '#805AD5',
        },
        pink: {
          450: '#ED64A6',
        },
        cyan: {
          450: '#00D4FF',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'float-reverse': 'floatReverse 8s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
        'rotate-glow': 'rotateGlow 10s linear infinite',
        'slide-in-up': 'slideInUp 0.6s ease-out',
        'fade-in': 'fadeIn 0.8s ease-out',
        'scale-in': 'scaleIn 0.5s ease-out',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'gradient-happy': 'linear-gradient(135deg, #FCD34D 0%, #F59E0B 50%, #FBBF24 100%)',
        'gradient-sad': 'linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #2563EB 100%)',
        'gradient-angry': 'linear-gradient(135deg, #F87171 0%, #EF4444 50%, #DC2626 100%)',
        'gradient-neutral': 'linear-gradient(135deg, #A78BFA 0%, #8B5CF6 50%, #7C3AED 100%)',
        'gradient-purple': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-pink': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'gradient-blue': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(102, 126, 234, 0.4)',
        'glow-lg': '0 0 40px rgba(102, 126, 234, 0.6)',
        'glow-pink': '0 0 20px rgba(240, 147, 251, 0.4)',
        'glow-blue': '0 0 20px rgba(79, 172, 254, 0.4)',
        'neon': '0 0 5px theme("colors.purple.400"), 0 0 20px theme("colors.purple.600")',
      },
    },
  },
  plugins: [],
}
export default config
