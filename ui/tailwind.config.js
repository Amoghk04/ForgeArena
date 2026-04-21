/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base: '#060610',
        panel: '#08081a',
        card: '#0c0c1e',
        'card-hover': '#111128',
        border: '#1c1c3a',
        'border-bright': '#2a2a55',
        neon: '#00d4ff',
        'neon-dim': '#0099cc',
        'neon-dark': '#002233',
        'neon-glow': 'rgba(0,212,255,0.15)',
        primary: '#e0e0ff',
        secondary: '#7080aa',
        muted: '#2e3452',
        'green-neon': '#00ff88',
        'orange-neon': '#ff9900',
        'red-neon': '#ff3366',
        'purple-neon': '#9955ff',
        'yellow-neon': '#ffee00',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        neon: '0 0 20px rgba(0,212,255,0.35), 0 0 60px rgba(0,212,255,0.1)',
        'neon-sm': '0 0 8px rgba(0,212,255,0.4)',
        'neon-card': '0 0 0 1px rgba(0,212,255,0.12), 0 4px 24px rgba(0,0,0,0.6)',
        'neon-inset': 'inset 0 1px 0 rgba(0,212,255,0.08)',
        'green': '0 0 12px rgba(0,255,136,0.4)',
        'red': '0 0 12px rgba(255,51,102,0.4)',
        'orange': '0 0 12px rgba(255,153,0,0.4)',
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s ease-in-out infinite',
        'scan': 'scan 3s linear infinite',
        'flicker': 'flicker 4s ease-in-out infinite',
        'slide-in': 'slide-in 0.3s ease-out',
        'fade-in': 'fade-in 0.4s ease-out',
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        flicker: {
          '0%, 95%, 100%': { opacity: 1 },
          '96%': { opacity: 0.8 },
          '97%': { opacity: 1 },
          '98%': { opacity: 0.7 },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-12px)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        'fade-in': {
          '0%': { opacity: 0, transform: 'translateY(8px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
