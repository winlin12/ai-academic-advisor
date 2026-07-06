/** Required for Tailwind to run at all — Next.js only applies the tailwindcss PostCSS
 * plugin when a postcss config declares it. See tailwind.config.ts. */
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};

export default config;
