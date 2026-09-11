import { useEffect, useState } from "react";

const KEY = "pfr-theme"; // "system" | "light" | "dark"

function apply(mode) {
  const root = document.documentElement;
  if (mode === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", mode);
}

export default function ThemeToggle() {
  const [mode, setMode] = useState(() => {
    try {
      return localStorage.getItem(KEY) || "system";
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    apply(mode);
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* private browsing / storage blocked — theme just won't persist */
    }
  }, [mode]);

  const next = { system: "light", light: "dark", dark: "system" };
  const labels = { system: "Auto", light: "Light", dark: "Dark" };

  return (
    <button
      className="theme-toggle"
      onClick={() => setMode(next[mode])}
      aria-label={`Colour theme: ${labels[mode]}. Click to change.`}
      title="Change colour theme"
      type="button"
    >
      {labels[mode]}
    </button>
  );
}
