import { useEffect, useState } from "react";

export type ThemeName = "dark" | "light";

const STORAGE_KEY = "memforge-dashboard-theme";

function initialTheme(): ThemeName {
  if (typeof window === "undefined") {
    return "dark";
  }
  const stored = window.localStorage?.getItem(STORAGE_KEY);
  if (stored === "dark" || stored === "light") {
    return stored;
  }
  return "dark";
}

/**
 * Dark is the default because this is an engineering tool used next to a code
 * editor; the choice is remembered locally and never sent anywhere.
 */
export function useTheme(): { theme: ThemeName; toggle: () => void } {
  const [theme, setTheme] = useState<ThemeName>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage?.setItem(STORAGE_KEY, theme);
    } catch {
      // A private-mode browser that refuses storage still gets the theme.
    }
  }, [theme]);

  return {
    theme,
    toggle: () => setTheme((current) => (current === "dark" ? "light" : "dark")),
  };
}
