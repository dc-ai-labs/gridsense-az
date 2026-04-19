# Practicality — who buys this, how they buy it, and how it fits their stack

A hackathon prototype only matters if a real buyer exists. This document lays
out who GridSense-AZ is for, why their workflow has room for it, how it would
integrate, what it would cost, and what regulatory constraints bound the
deployment.

## Target customer

Mid-to-large **distribution utilities** operating radial feeders in hot-climate
service territories. Primary candidates in the Southwest:

- **Arizona Public Service (APS)** — 1.3 M customers, 36,000 mi² service
  territory, summer peak ~8 GW. The hackathon sponsor and our calibration
  target.
- **Salt River Project (SRP)** — 1.1 M customers, Phoenix metro.
- **Tucson Electric Power (TEP)** — 450 k customers, Tucson metro.
- Out-of-state analogues: NV Energy, El Paso Electric, PG&E's Central Valley
  distribution, FPL's inland feeders.

### Buyer personas

| Role | What they do today | What GridSense-AZ changes |
|---|---|---|
| Distribution dispatcher (control centre, 24×7) | Watches SCADA, dispatches field crews, manages load switching in real time | Gives them a 24 h lookahead of per-bus stress with uncertainty bands — lets them pre-position crews and stage switching plans before events, not after |
| DMS (distribution-management-system) engineer | Owns the software: ADMS vendor, CIM model, topology. Evaluates new tooling | Plugs into existing CIM pipeline; sits *alongside* ADMS, not replacing it |
| DER-integration lead | Plans EV-charging tariffs, rooftop-solar hosting-capacity studies, battery-siting | Uses the EV-surge scenario to stress-test tariff proposals; uses risk-leaderboard to pick battery-hosting candidate buses |
| Asset-management planner | 5–10 year forecast of conductor upgrades | Uses the stress-hour histogram to prioritise which feeder laterals see the most 100%-of-rating hours |

## Why they would buy

There is a visible **workflow gap** between the two systems a utility already
owns:

1. **Load Forecasting System (LFS)** — produces **system-wide** day-ahead and
   week-ahead forecasts. Coarse, aggregated, no per-bus breakdown, no physics
   awareness. Typically provided by the ISO/RTO (or in-house).
2. **SCADA / ADMS** — real-time telemetry, sub-second situational awareness.
   No prediction. Looks backward from "now."

**Nothing** off-the-shelf does **per-bus, day-ahead, physics-checked, with
uncertainty**. GridSense-AZ lives in that seat: day-ahead operations planning.
The closest analogue is manual Excel work by a senior engineer who happens to
know which feeders get hot and eyeballs the PRIM load outputs.

Concrete value propositions to put in a quarterly report:

- **Crew pre-positioning.** If risk-leaderboard flags a 0.85-risk bus 14 h out,
  a line crew can be staged within response distance at the start of shift
  rather than recalled from lunch.
- **Demand-response targeting.** Existing DR programmes call all opted-in
  customers. Per-bus forecasts let you call **only** the ones on at-risk
  laterals, preserving DR budget for when it matters.
- **Capital-deferral decisions.** A conductor upgrade is $100k–$1M / mi. A
  stress-hour histogram across the year lets the planner defer upgrades where
  the 95th-percentile loading is actually fine.

## Integration path

We do not ask the utility to rip out anything they already own. Integration
points in order of "how hard":

### Easy — data in
- **Weather**: NOAA / NWS (already public).
- **Demand history**: the utility's AMI / SCADA historian (typically OSIsoft
  PI or a similar time-series DB). REST or OPC-HDA export, hourly aggregates.
- **Topology**: CIM XML or DSS master. APS uses an OSI ADMS → CIM is native.

### Medium — model in
- GridSense-AZ is a single PyTorch checkpoint + a thin Python service. Runs
  on-prem, inside the utility's OT DMZ. Inference is CPU-friendly (60k params).
- A GPU is only needed for retraining — ~9 min on an A100, once per season
  is sufficient.

### Easy — output out
- **REST**: `GET /forecast/{feeder_id}` returns the JSON schema we already
  ship to the dashboard (p10/p50/p90 per bus per hour + OpenDSS snapshot).
- **CSV**: the same data flattened, for engineers who drop everything into
  Excel.
- **CIM**: roadmap item. Utilities running ADMS/DMS in IEC 61968/61970 want
  CIM-compliant message payloads so our forecast can flow into their DMS
  dashboards natively.
- **ADMS plug-in**: longer term — a SCL-compatible adapter for the major
  ADMS products.

### Deployment topology

```
 Utility SCADA / AMI            Utility weather feed
         │                           │
         ▼                           ▼
  time-series historian       NOAA / NWS cache
         │                           │
         └──────────┬────────────────┘
                    ▼
          GridSense-AZ service
           (PyTorch + OpenDSS)
              VPC / on-prem
                    │
                    ▼
         REST JSON → DMS / ADMS
         CSV / PDF → ops email
         Dashboard → control room
```

SaaS is **not** an option for most OT data; utilities want VPC-hosted or
on-prem. We ship a Docker image; hosting is the utility's call.

## Pricing model (sketch)

Reference points from publicly stated utility-software pricing:

| Product category | Typical annual licence / feeder |
|---|---|
| Narrow-use analytics (e.g. DLR tools) | $10k – $30k |
| ML forecasting SaaS (AutoGrid, Urbint, etc.) | $25k – $150k |
| Full ADMS suite (OSI, Schneider, GE) | $1 M – $10 M enterprise |

Our positioning: **per-feeder annual licence, $40k – $80k** tiered on
accuracy guarantees (MAE-target SLAs) and number of scenario transforms
included. Large-utility enterprise agreement covering 50–500 feeders is the
natural expansion motion after a one-feeder pilot.

Costs to us: hosting is nil (on-prem at the utility). Marginal cost per
deployment is support hours; the model retrain is GPU-hours at commodity
cloud rates (~$3/h × 10 min ≈ $0.50 per seasonal retrain).

## Regulatory

Utility software in North America must navigate at minimum:

- **NERC CIP-002 through CIP-014** — Critical Infrastructure Protection
  standards. CIP-002 is the BES Cyber System categorisation; CIP-003/005/007
  govern access control, electronic perimeter, and system hardening; CIP-011
  covers information protection; CIP-013 is supply-chain. A GridSense-AZ
  deployment inside the OT DMZ inherits the utility's existing CIP posture
  (our service is a read-only consumer of forecasts + writer of JSON).
- **FERC Order 881** — Ambient-Adjusted Ratings (AAR). Requires transmission
  line ratings to use hourly ambient temperature. Our **thermal overload
  indicator is rating-aware but static**; a production release would add
  dynamic line rating (ambient + wind). Flagged in `PHYSICS.md`.
- **State PUCs** — Arizona Corporation Commission for APS/TEP; they approve
  rate cases and capital plans, not tooling purchases directly, but any
  tool whose output affects capital deferral or DR dispatch enters the record.
- **NIST Cybersecurity Framework / NIST 800-53** — utilities reference these
  for control-system security. No blockers but worth listing for the RFP.

Compliance posture we ship with: read-only integration, no actuation, no
control-loop authority. The tool is advisory — the human dispatcher acts.
That keeps GridSense-AZ firmly outside the authority-to-operate burden of
control software.

## Competitive landscape

| Vendor | Product | Gap vs GridSense-AZ |
|---|---|---|
| **Oracle** | NMS / Opower | System-level forecasting, no per-bus ML + physics loop |
| **GE** | GridOS, PowerOn Advantage | Rules-based, weak on learned per-bus forecasting |
| **Schneider Electric** | ADMS | Great for real-time, no day-ahead ML forecast |
| **Siemens** | Spectrum Power | Similar — strong on execution, weak on ML forecast |
| **AutoGrid (Schneider)** | Flex / DRMS | DR-focused, system-level, not per-bus |
| **Urbint** | Lens | Safety + vegetation focus, not load |
| **Utilidata** | AMI analytics | Reactive, not predictive |

We are not trying to replace any of them. We slot **between** the LFS and
ADMS — a day-ahead operations-planning layer with the ML-plus-physics loop
none of the incumbents offer.

## Who this is NOT for

- **ISO / RTO wholesale market forecasting** — different scale (GWs,
  thousands of nodes), different economics (LMPs, bids), different regulators.
  CAISO / MISO / PJM already run their own.
- **Residential building energy management (BEMS)** — we are utility-side,
  not customer-side.
- **Utility-scale solar siting / generation interconnection studies** — PSS/E
  / PSLF / PowerWorld territory. Transmission, not distribution.
- **Load research / DSM programme analytics** — different tool, different
  user, different time horizon.

## What's left to close before a real pilot

1. **Real AMI ingest path** (currently synthetic disaggregation).
2. **CIM message-bus adapter** for DMS integration.
3. **SLA wording** on forecast accuracy and uptime for the contract template.
4. **CIP-003/011 documentation package** — paperwork, not engineering.
5. **Dynamic Line Rating** module to satisfy FERC 881-derived expectations.
