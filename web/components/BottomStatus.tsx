export default function BottomStatus() {
  return (
    <footer className="h-8 bg-surface-container-low border-t border-outline-variant flex items-center justify-between px-4 text-[9px] font-mono uppercase tracking-tighter text-on-surface-variant shrink-0">
      <div className="flex gap-4">
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 bg-primary" /> SYSTEM_NOMINAL_AUTO
        </span>
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[10px]">database</span>{" "}
          REPL_LAG: 42ms
        </span>
      </div>
      <div className="flex gap-4">
        <span className="text-primary">PHX_CENTER_WEST_NODE: ACTIVE</span>
        <span>SECURE_SHELL_TLS_1.3</span>
      </div>
    </footer>
  );
}
