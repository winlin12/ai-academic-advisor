import type { Config } from "tailwindcss";

// NOTE: this file (plus postcss.config.mjs) is what makes Tailwind actually compile.
// The app shipped for weeks with neither, so every `className` silently rendered as
// unstyled browser defaults — if the UI ever regresses to "plain HTML" again, check
// these two files first.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Purdue brand palette (https://marcom.purdue.edu/toolbox/colors/)
        boiler: {
          black: "#000000",
          gold: "#CFB991", // Boilermaker Gold
          field: "#DDB945",
          dust: "#EBD99F",
          aged: "#8E6F3E",
          steel: "#555960",
          gray: "#6F727B",
        },
      },
      fontFamily: {
        display: ["Barlow Condensed", "Oswald", "Arial Narrow", "sans-serif"],
        body: ["Space Grotesk", "Avenir Next", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
