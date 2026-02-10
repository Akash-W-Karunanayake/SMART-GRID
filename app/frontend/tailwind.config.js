/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Power system theme colors
        grid: {
          primary: '#1e40af',    // Deep blue - main power
          secondary: '#059669',  // Green - renewable/solar
          warning: '#d97706',    // Amber - caution
          danger: '#dc2626',     // Red - fault/violation
          neutral: '#6b7280',    // Gray - inactive
        },
        // Voltage level indicators
        voltage: {
          high: '#ef4444',      // Red - overvoltage
          normal: '#22c55e',    // Green - normal
          low: '#f59e0b',       // Amber - undervoltage
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flow': 'flow 2s linear infinite',
      },
      keyframes: {
        flow: {
          '0%': { strokeDashoffset: '0' },
          '100%': { strokeDashoffset: '-20' },
        }
      }
    },
  },
  plugins: [],
}
