import "./globals.css";

export const metadata = {
  title: "AI Academic Advisor",
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
        {children}
      </body>
    </html>
  );
}
