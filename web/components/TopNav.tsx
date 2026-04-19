export default function TopNav() {
  return (
    <header className="bg-[#111319] text-[#4fdbc8] font-headline tracking-tight top-0 h-14 border-b border-[#3c4947] flex justify-between items-center w-full px-6 z-50 shrink-0">
      <div className="flex items-center gap-4">
        <span className="text-xl font-bold tracking-tighter text-[#4fdbc8]">
          GRIDSENSE-AZ
        </span>
        <div className="h-4 w-px bg-outline-variant mx-2" />
        <nav className="hidden md:flex gap-6 text-xs uppercase tracking-widest font-mono">
          <a
            className="text-[#4fdbc8] border-b-2 border-[#4fdbc8] h-14 flex items-center px-2"
            href="#"
          >
            GRID_OVERVIEW
          </a>
          <a
            className="text-slate-500 hover:bg-[#4fdbc8]/10 transition-colors duration-75 h-14 flex items-center px-2"
            href="#"
          >
            LIVE_NODES
          </a>
          <a
            className="text-slate-500 hover:bg-[#4fdbc8]/10 transition-colors duration-75 h-14 flex items-center px-2"
            href="#"
          >
            FAULT_LOGS
          </a>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex gap-2">
          <button
            aria-label="notifications"
            className="p-2 hover:bg-[#4fdbc8]/10 transition-colors duration-75"
          >
            <span className="material-symbols-outlined text-sm">
              notifications
            </span>
          </button>
          <button
            aria-label="settings"
            className="p-2 hover:bg-[#4fdbc8]/10 transition-colors duration-75"
          >
            <span className="material-symbols-outlined text-sm">settings</span>
          </button>
          <button
            aria-label="topology"
            className="p-2 hover:bg-[#4fdbc8]/10 transition-colors duration-75"
          >
            <span className="material-symbols-outlined text-sm">
              account_tree
            </span>
          </button>
        </div>
        <div className="flex items-center gap-3 ml-4">
          <span className="text-[10px] font-mono text-right leading-none hidden lg:block">
            OPERATOR_ID
            <br />
            <span className="text-primary">AZ_4922_SYS</span>
          </span>
          <div className="w-8 h-8 bg-surface-container-highest border border-outline-variant flex items-center justify-center">
            <span className="material-symbols-outlined text-sm text-primary">
              person
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
