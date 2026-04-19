# GridSense-AZ Dashboard (web/)

Next.js 14 App Router tactical-ops dashboard for the GridSense-AZ operator console.
Consumes static JSON forecasts from `public/data/forecasts/*.json` produced by
`scripts/precompute_forecasts.py` at the Python repo root.

## Dev

```bash
cd web
pnpm install
pnpm dev     # http://localhost:3000
```

## Build

```bash
pnpm build
pnpm start
```

## Deploy (Vercel)

```bash
vercel --prod --yes            # from inside web/
```

## Keyboard shortcuts

- `B` — baseline scenario
- `H` — +10 deg F heat scenario
- `E` — EV surge scenario

## Data contract

All forecast JSON shapes live in `lib/types.ts`. Components read from
`ScenarioContext` (see `lib/context.tsx`); a single `active` scenario key
(`"baseline" | "heat" | "ev"`) selects which pre-loaded forecast drives the UI.

Sample stubs ship in `public/data/forecasts/*.json` so the dashboard builds and
runs standalone before Slice B (`scripts/precompute_forecasts.py`) lands real
NWS-driven data.
