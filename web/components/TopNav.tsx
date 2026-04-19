"use client";

import { useEffect, useRef, useState } from "react";
import { useScenario } from "@/lib/context";

type DayPreset = "tomorrow" | "replay_2023_07_15";

const DAY_PRESET_KEY = "gridsense:dayPreset";

function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "UNKNOWN";
  const diffMs = Date.now() - then;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "JUST NOW";
  if (diffMin < 60) return `${diffMin}M AGO`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}H AGO`;
  try {
    return new Date(iso)
      .toLocaleDateString("en-US", {
        timeZone: "America/Phoenix",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      })
      .replace(/\//g, "-");
  } catch {
    return iso.slice(0, 10);
  }
}

function formatPhoenixDate(iso: string): string {
  try {
    // en-CA gives YYYY-MM-DD format directly.
    return new Date(iso).toLocaleDateString("en-CA", {
      timeZone: "America/Phoenix",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  } catch {
    return iso.slice(0, 10);
  }
}

function sourceDotClass(src: "live" | "replay" | "synthetic"): string {
  if (src === "live") return "bg-primary";
  if (src === "replay") return "bg-secondary";
  return "bg-error";
}

function sourceLabel(src: "live" | "replay" | "synthetic"): string {
  if (src === "live") return "NWS LIVE";
  if (src === "replay") return "NWS REPLAY";
  return "SYNTHETIC";
}

export default function TopNav() {
  const { generatedAt, current } = useScenario();

  const [provenanceOpen, setProvenanceOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [preset, setPreset] = useState<DayPreset>("tomorrow");
  const [toast, setToast] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const provenanceRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Hydrate preset from sessionStorage.
  useEffect(() => {
    try {
      const stored = window.sessionStorage.getItem(DAY_PRESET_KEY);
      if (stored === "tomorrow" || stored === "replay_2023_07_15") {
        setPreset(stored);
      }
    } catch {
      /* sessionStorage may be unavailable */
    }
  }, []);

  // Tick for relative-time label so "11H AGO" stays fresh.
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, []);
  // Reference `now` so linters don't complain about unused state while still
  // forcing a re-render each minute.
  void now;

  // Close popovers on Esc + click-outside.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (helpOpen) setHelpOpen(false);
        if (provenanceOpen) setProvenanceOpen(false);
        if (dropdownOpen) setDropdownOpen(false);
      }
    };
    const onClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (provenanceOpen && provenanceRef.current && !provenanceRef.current.contains(t)) {
        setProvenanceOpen(false);
      }
      if (dropdownOpen && dropdownRef.current && !dropdownRef.current.contains(t)) {
        setDropdownOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, [helpOpen, provenanceOpen, dropdownOpen]);

  // Auto-dismiss toast.
  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(id);
  }, [toast]);

  const handleSelectPreset = (next: DayPreset) => {
    setDropdownOpen(false);
    if (next === "replay_2023_07_15") {
      setToast(
        "Replay mode requires precompute script — coming soon. Run `python scripts/precompute_forecasts.py --replay 2023-07-15` to enable.",
      );
      // Do NOT persist replay — stay on tomorrow so data doesn't look broken.
      return;
    }
    setPreset(next);
    try {
      window.sessionStorage.setItem(DAY_PRESET_KEY, next);
    } catch {
      /* noop */
    }
  };

  // Forecast target date: peak-hour quantile timestamp in Phoenix TZ.
  const peakRow = current.quantiles.find((q) => q.hour === current.feeder_rollup.peak_hour);
  const forecastDate = peakRow ? formatPhoenixDate(peakRow.ts) : formatPhoenixDate(generatedAt.iso);
  const relative = formatRelative(generatedAt.iso);
  const gitShort = generatedAt.git_sha?.slice(0, 7) ?? "unknown";
  const src = generatedAt.nws_source;

  return (
    <>
      <header className="bg-[#111319] text-[#4fdbc8] font-headline tracking-tight top-0 h-14 border-b border-[#3c4947] flex justify-between items-center w-full px-4 md:px-6 z-50 shrink-0 gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-xl font-bold tracking-tighter text-[#4fdbc8] whitespace-nowrap">
            GRIDSENSE-AZ
          </span>
          <div className="h-4 w-px bg-outline-variant mx-1 hidden sm:block" />

          {/* DATA PROVENANCE pill */}
          <div ref={provenanceRef} className="relative hidden sm:block">
            <button
              type="button"
              onClick={() => setProvenanceOpen((v) => !v)}
              aria-expanded={provenanceOpen}
              aria-label="Data provenance details"
              className="flex items-center gap-2 px-2 md:px-3 h-8 border border-outline-variant bg-surface-container hover:bg-surface-container-high transition-colors duration-75 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant"
            >
              <span className={`inline-block w-2 h-2 ${sourceDotClass(src)} blinking`} />
              <span className="text-primary whitespace-nowrap">{sourceLabel(src)}</span>
              <span className="text-outline hidden lg:inline">·</span>
              <span className="hidden lg:inline whitespace-nowrap">FORECAST {forecastDate}</span>
              <span className="text-outline hidden xl:inline">·</span>
              <span className="hidden xl:inline whitespace-nowrap">GEN {relative}</span>
              <span className="material-symbols-outlined text-[14px] leading-none text-outline">
                info
              </span>
            </button>
            {provenanceOpen && (
              <div className="absolute left-0 top-[calc(100%+4px)] z-50 w-80 bg-surface-container-high border border-primary p-4 shadow-2xl text-[11px] font-mono">
                <div className="text-primary uppercase tracking-widest font-bold mb-3 text-xs">
                  DATA_PROVENANCE
                </div>
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-on-surface-variant">
                  <dt className="text-outline uppercase">source</dt>
                  <dd className="text-on-surface">{sourceLabel(src)} ({src})</dd>
                  <dt className="text-outline uppercase">forecast</dt>
                  <dd className="text-on-surface">{forecastDate} (America/Phoenix)</dd>
                  <dt className="text-outline uppercase">generated</dt>
                  <dd className="text-on-surface break-all">{generatedAt.iso}</dd>
                  <dt className="text-outline uppercase">horizon</dt>
                  <dd className="text-on-surface">{generatedAt.hours_forecast}h</dd>
                  <dt className="text-outline uppercase">git_sha</dt>
                  <dd className="text-on-surface font-mono">{gitShort}</dd>
                  <dt className="text-outline uppercase">weather</dt>
                  <dd className="text-on-surface">
                    {current.weather.source} · peak {current.weather.peak_temp_f.toFixed(1)}°F @ h{current.weather.peak_hour}
                  </dd>
                </dl>
              </div>
            )}
          </div>

          <nav className="hidden xl:flex gap-6 text-xs uppercase tracking-widest font-mono ml-2">
            <a
              className="text-[#4fdbc8] border-b-2 border-[#4fdbc8] h-14 flex items-center px-2"
              href="#"
            >
              GRID_OVERVIEW
            </a>
          </nav>
        </div>

        <div className="flex items-center gap-2 md:gap-3">
          {/* DAY PRESET dropdown */}
          <div ref={dropdownRef} className="relative hidden md:block">
            <button
              type="button"
              onClick={() => setDropdownOpen((v) => !v)}
              aria-expanded={dropdownOpen}
              aria-haspopup="listbox"
              className="flex items-center gap-2 h-8 px-3 border border-outline-variant bg-surface-container hover:bg-surface-container-high transition-colors duration-75 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant"
            >
              <span className="text-outline">DAY:</span>
              <span className="text-primary whitespace-nowrap max-w-[14ch] truncate">
                {preset === "tomorrow" ? "TOMORROW (LIVE NWS)" : "2023-07-15 REPLAY"}
              </span>
              <span className="material-symbols-outlined text-[14px] leading-none text-outline">
                expand_more
              </span>
            </button>
            {dropdownOpen && (
              <ul
                role="listbox"
                className="absolute right-0 top-[calc(100%+4px)] z-50 w-72 bg-surface-container-high border border-primary shadow-2xl text-[11px] font-mono"
              >
                <li
                  role="option"
                  aria-selected={preset === "tomorrow"}
                  onClick={() => handleSelectPreset("tomorrow")}
                  className={`px-3 py-2 cursor-pointer hover:bg-primary/10 flex items-start gap-2 ${
                    preset === "tomorrow" ? "text-primary" : "text-on-surface"
                  }`}
                >
                  <span className="text-[10px] mt-0.5">{preset === "tomorrow" ? "▸" : " "}</span>
                  <span className="flex-1">
                    <span className="block uppercase tracking-widest">TOMORROW (LIVE NWS)</span>
                    <span className="block text-outline text-[10px] mt-0.5">
                      Forecast driven by live NWS hourly pull
                    </span>
                  </span>
                </li>
                <li
                  role="option"
                  aria-selected={preset === "replay_2023_07_15"}
                  onClick={() => handleSelectPreset("replay_2023_07_15")}
                  className="px-3 py-2 cursor-pointer hover:bg-primary/10 flex items-start gap-2 text-on-surface border-t border-outline-variant"
                >
                  <span className="text-[10px] mt-0.5"> </span>
                  <span className="flex-1">
                    <span className="block uppercase tracking-widest">
                      2023-07-15 (PHX 119°F RECORD REPLAY)
                    </span>
                    <span className="block text-outline text-[10px] mt-0.5">
                      Heat-wave day. Requires replay precompute.
                    </span>
                  </span>
                </li>
              </ul>
            )}
          </div>

          {/* HELP button */}
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            aria-label="Keyboard shortcuts and help"
            className="flex items-center gap-1 h-8 px-3 border border-outline-variant bg-surface-container hover:bg-surface-container-high transition-colors duration-75 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant"
          >
            <span className="material-symbols-outlined text-[14px] leading-none text-primary">
              help
            </span>
            <span className="text-primary hidden md:inline">HELP</span>
            <span className="text-outline hidden md:inline">[?]</span>
          </button>

          <div className="flex items-center gap-3 ml-1 md:ml-2">
            <span className="text-[10px] font-mono text-right leading-none hidden lg:block">
              OPERATOR_ID
              <br />
              <span className="text-primary">AZ_4922_SYS</span>
            </span>
            <div className="w-8 h-8 bg-surface-container-highest border border-outline-variant flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-sm text-primary">
                person
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* HELP modal */}
      {helpOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) setHelpOpen(false);
          }}
          role="dialog"
          aria-modal="true"
          aria-label="Keyboard shortcuts"
        >
          <div className="w-full max-w-lg mx-4 bg-surface-container-high border border-primary shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3 border-b border-outline-variant">
              <span className="text-primary font-headline font-bold uppercase tracking-widest text-sm">
                GRIDSENSE-AZ · QUICK REFERENCE
              </span>
              <button
                type="button"
                onClick={() => setHelpOpen(false)}
                aria-label="Close help"
                className="p-1 hover:bg-primary/10 transition-colors"
              >
                <span className="material-symbols-outlined text-base text-on-surface-variant">
                  close
                </span>
              </button>
            </div>
            <div className="p-5 space-y-5 text-xs font-mono">
              <section>
                <div className="text-outline uppercase tracking-widest text-[10px] mb-2">
                  KEYBOARD
                </div>
                <ul className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-on-surface">
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[2.5ch] text-center">
                      B
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">baseline scenario</li>
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[2.5ch] text-center">
                      H
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">heat +10°F scenario</li>
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[2.5ch] text-center">
                      E
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">EV surge scenario</li>
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[2.5ch] text-center">
                      C
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">cycle compare overlay</li>
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[2.5ch] text-center">
                      R
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">reset what-if sliders</li>
                  <li className="flex items-center gap-2">
                    <kbd className="inline-block px-2 py-0.5 border border-outline-variant bg-surface-container text-primary min-w-[3.5ch] text-center">
                      Esc
                    </kbd>
                  </li>
                  <li className="text-on-surface-variant">clear map focus / close overlay</li>
                </ul>
              </section>
              <section>
                <div className="text-outline uppercase tracking-widest text-[10px] mb-2">
                  MOUSE
                </div>
                <ul className="space-y-1.5 text-on-surface-variant">
                  <li>
                    <span className="text-primary">click bus row</span> on leaderboard · focus
                    that bus on map
                  </li>
                  <li>
                    <span className="text-primary">click overload row</span> on physics-check ·
                    highlight feeder line
                  </li>
                  <li>
                    <span className="text-primary">drag sliders</span> in mission control ·
                    custom what-if scenario
                  </li>
                </ul>
              </section>
              <div className="text-[10px] text-outline uppercase tracking-widest pt-1 border-t border-outline-variant">
                Close: Esc · click outside · ✕
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div
          role="status"
          className="fixed bottom-14 left-1/2 -translate-x-1/2 z-[120] max-w-md mx-4 px-4 py-3 bg-surface-container-high border border-secondary text-xs font-mono text-on-surface shadow-2xl"
        >
          <div className="flex items-start gap-3">
            <span className="material-symbols-outlined text-base text-secondary leading-none mt-0.5">
              info
            </span>
            <span className="flex-1">{toast}</span>
            <button
              type="button"
              onClick={() => setToast(null)}
              aria-label="Dismiss toast"
              className="text-outline hover:text-on-surface"
            >
              <span className="material-symbols-outlined text-sm leading-none">close</span>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
