import BottomStatus from "@/components/BottomStatus";
import ForecastRibbon from "@/components/ForecastRibbon";
import MissionControl from "@/components/MissionControl";
import ModelMetrics from "@/components/ModelMetrics";
import PhysicsCheck from "@/components/PhysicsCheck";
import RiskLeaderboard from "@/components/RiskLeaderboard";
import TacticalMap from "@/components/TacticalMap";
import TopNav from "@/components/TopNav";
import { ScenarioProvider } from "@/lib/context";

export default function Page() {
  return (
    <ScenarioProvider>
      <TopNav />
      <main className="flex-1 grid grid-cols-[280px_1fr_320px] overflow-hidden min-h-0">
        <aside className="etched-r overflow-y-auto bg-surface">
          <MissionControl />
        </aside>
        <section className="flex flex-col etched-r min-h-0">
          <ForecastRibbon />
          <TacticalMap />
        </section>
        <aside className="overflow-y-auto flex flex-col bg-surface min-h-0">
          <div className="flex-shrink-0">
            <ModelMetrics />
          </div>
          <div className="flex-shrink-0">
            <RiskLeaderboard />
          </div>
          <div className="flex-shrink-0">
            <PhysicsCheck />
          </div>
        </aside>
      </main>
      <BottomStatus />
    </ScenarioProvider>
  );
}
