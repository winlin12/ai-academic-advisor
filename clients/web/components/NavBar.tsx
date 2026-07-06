"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Planner" },
  { href: "/admin", label: "Database" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20">
      <nav className="border-b border-[var(--stroke)] bg-black/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 md:px-8">
          <Link href="/" className="flex items-center gap-3">
            <span className="font-display flex h-9 w-9 items-center justify-center rounded bg-gradient-to-b from-[var(--gold-soft)] to-[var(--gold-bright)] text-xl font-bold text-black">
              B
            </span>
            <span className="font-display text-2xl font-semibold uppercase tracking-wide">
              <span className="text-[var(--gold)]">Boiler</span>
              <span className="text-[var(--ink)]">Advisor</span>
            </span>
          </Link>

          <div className="flex items-center gap-1 md:gap-2">
            {links.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={
                    active
                      ? "rounded-lg bg-[rgba(207,185,145,0.14)] px-3 py-2 text-sm font-semibold text-[var(--gold)]"
                      : "rounded-lg px-3 py-2 text-sm font-medium text-[var(--muted)] transition hover:bg-[rgba(207,185,145,0.08)] hover:text-[var(--ink)]"
                  }
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>
      <div className="boiler-stripe" />
    </header>
  );
}
