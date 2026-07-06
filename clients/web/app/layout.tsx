import "./globals.css";
import NavBar from "@/components/NavBar";

export const metadata = {
  title: "BoilerAdvisor",
  description:
    "Local-first Purdue academic planning: build, edit, and validate semester-by-semester degree plans.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="mesh-bg" />
        <NavBar />
        {children}
        <footer className="mx-auto max-w-7xl px-4 pb-10 pt-16 text-center text-xs text-[var(--muted)] md:px-8">
          BoilerAdvisor is a planning tool, not official academic advising — verify
          decisions with your university. Not affiliated with Purdue University.
        </footer>
      </body>
    </html>
  );
}
