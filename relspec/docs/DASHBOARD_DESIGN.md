# relspec dashboard — product design

The platform stores honest physics; this document designs the product on
top of it. Reference points: Augury's health-tier machine cards and
insight feed, Tractian's asset tree, prescriptive findings inbox and
mobile-first cards — borrowed where they fit a system whose spine is
different from both: every claim on screen traces to a spectrum, a
pattern, or an event that the user can open.

Companion mockup: `viz/dashboard_mockup.html` (Fleet, Asset, Mobile).

---

## 1. Design principles

1. **Evidence or it doesn't ship.** Every health state, finding and alarm
   links to the frames that caused it. No black-box "AI score".
2. **The machine states, the human diagnoses.** relspec computes change,
   levels, and pattern behaviour; *people* attach meaning as findings.
   The product makes that collaboration first-class instead of pretending
   diagnosis is automatic.
3. **Server computes, clients render.** Health, z-scores, rollups are API
   outputs with stated formulas — the mobile app and desktop suite can
   never disagree.
4. **Honesty widgets are features.** Showing where raw truth exists vs
   codec reconstruction (and the measured error at every truth point) is
   a selling point, not a confession.

## 2. Health model

### 2.1 Per-sensor health

Four tiers — **Healthy · Monitor · Alert · Critical** — computed
server-side per sensor from signals we already store, each with a stated
contribution (shown on hover, always inspectable):

| signal | source | maps to |
|---|---|---|
| ISO velocity zone | vel_rms vs class limits (per machine_type) | floor state |
| Gate state | score vs s_lo/s_hi, CUSUM drift, open change events | change state |
| Pattern z (§3) | max z over verified fault-relevant tracks | pattern state |
| Trend slope | trailing 30-frame slope of vel/acc/env RMS and top track energies | direction |

Combination: `tier = max(floor, change, pattern)` with one-step
**hysteresis** (a tier only relaxes after N=5 consecutive quieter frames)
so cards don't flap. `direction` renders as ▲▼ beside the tier, never
changes it.每 frame stores a `health` row: (tier, drivers[], since_ts) —
"drivers" is the list of contributing signals with values, so the UI can
say *why* in one line: "Alert — BPFO track z 6.2, vel 5.8 mm/s (zone C)".

### 2.2 Rollups

Component = worst of its sensors. Asset = worst of components. Plant =
distribution, not just worst: the plant card shows a **health donut**
(Augury-style) + count per tier, because "1 critical of 120" and "40
monitor of 120" are different mornings. Tree navigation (Tractian-style
left rail on desktop, drill-down cards on mobile) shows tier chips at
every level, aggregated live from `sensor_latest`.

### 2.3 Health timeline

Per asset: a horizontal band chart of tier over time (colored strip),
with event/finding markers on it. This is the anchor visual of the asset
page — "what happened to this machine" in one glance.

## 3. Per-pattern energy z-score

The question "is THIS pattern misbehaving relative to ITSELF" — the
per-bin z waterfall generalised to the object the analyst actually
tracks.

- **Baseline**: per (sensor, rail, track): robust median + MAD of
  `10·log10(energy)` over the **baseline window** = frames since the last
  `baseline_reset` event (or sensor start), excluding frames the gate
  held in change, minimum 10 observations before z is reported (before
  that: "learning").
- **Score**: `z = (E_dB − med) / (1.4826·MAD)`, clamped ±10, recomputed
  at ingest, stored on the pattern row (`z REAL`), baseline stats in a
  new `pattern_baseline` table (sensor, rail, track_id, n, med, mad,
  window_start). Absence scoring: a track that was ≥1% share in baseline
  and vanishes for 3+ frames gets `z = −10` flagged "vanished" (a lost
  comb can be a repair — or a broken sensor).
- **Visuals**:
  - **Pattern health matrix** (asset/sensor page): tracks × time heatmap
    coloured by z (diverging, ±6), track labels with kind chip + key +
    kinematic name when the component's kinematics resolve it (order
    3.58 → "BPFO (SKF 6205)"). This is the fastest "what changed"
    surface in the product.
  - Ledger table gains a **z column** with the same diverging pill.
  - Health model consumes `max z` over verified tracks (§2.1).

## 4. Collaboration: findings & events

Single-passphrase workspaces have no accounts, so identity is
**self-declared**: a display name kept client-side, stamped on
everything the person writes. Stated honestly in the UI ("names are
labels, not authentication").

- **Finding** (the Tractian-style insight, human- or rule-created):
  severity (info/monitor/alert/critical), free title, fault taxonomy tag
  (bearing-outer, bearing-inner, imbalance, misalignment, looseness,
  lubrication, electrical, other), body, **evidence links** (acq ids,
  track ids, frequency ranges — chosen by clicking in the explorer:
  "attach this frame / this track to finding…"), status
  **open → acknowledged → closed** with timestamps + names, and a
  comment thread.
- **Events timeline** unifies: gate changes (machine-created), findings
  (human), maintenance notes, `baseline_reset`, `data_cutoff`, sensor
  edits. One lane on every timeline; filterable inbox per plant
  (Augury-style feed: newest insights first, tier-colored, with
  ack buttons).
- API: `POST/GET /v1/findings`, `POST /v1/findings/{id}/comments`,
  `PATCH /v1/findings/{id}` (status), all appending to `event` so the
  history is append-only; edits create revisions, never rewrites.

## 5. Baseline management: reset & cutoff

Machines get repaired, rebuilt, re-tasked; old normal becomes noise.

- **Reset baseline** (`POST /v1/sensors/{id}/reset-baseline {reason}`):
  forces a codec anchor on next ingest, re-initialises the gate baseline,
  restarts every pattern baseline window (§3), stamps a `baseline_reset`
  event. The health tier goes to "learning" until warm again. Shown on
  all timelines as a labelled vertical marker — analysis views default to
  windows that do not cross it.
- **Data cutoff** (`POST /v1/sensors/{id}/cutoff {before_ts, reason}`):
  soft-archives history before the date: excluded from baselines, trends
  and health by default (a "show archived" toggle reveals it, dimmed);
  raw waveforms before cutoff become eligible for early retention.
  Recorded as `data_cutoff` event. Nothing is deleted by the button —
  deletion stays an explicit retention policy.
- Both actions require typing the sensor code to confirm (destructive-ish
  UX), and both are events so the "why does the baseline start here?"
  question always has an answer on the timeline.

## 6. Raw vs codec: the confidence timeline

The compression claim becomes a visible, self-auditing feature:

- **Snapshot lane** on every sensor timeline: ◆ where a full raw
  waveform exists (colored by sig_reason: first/baseline/event/pinned),
  · for codec-only frames. The ratio is stated plainly:
  "37 raw snapshots · 1,058 codec frames · 322× average".
- **Truth-point audit**: at every ◆ the server recomputes the spectrum
  from raw and compares against the decoded frame — med/p95/max dB error
  badges on the snapshot popover, and a small **confidence trend** chart
  (error at snapshots over time) on the sensor page. If codec error ever
  drifts at truth points, the product shows it before a customer asks.
  (Endpoint: `GET /v1/acquisitions/{id}/audit` — recompute, compare,
  cache; plus rolled-up `GET /v1/sensors/{id}/confidence`.)
- **Snapshot browser**: click ◆ → overlay raw-derived spectrum, binned,
  decoded (the explorer inspector, fed by truth), plus time-waveform and
  envelope views (desktop suite, §8).
- **Storage parametrization** (the "store more" knob): per workspace and
  per sensor overrides, editable in settings with a live cost estimate:
  `baseline_days` (7 → e.g. 1 for daily snapshots), `event_budget_per_week`,
  `retention_days`, and `pin` on individual acquisitions (subject to
  budget). The estimate uses measured numbers: "daily snapshots on this
  sensor ≈ 13 MB/year; weekly ≈ 1.8 MB/year". Changing parameters is an
  event too.

## 7. Mobile (card-first)

Not a shrunken desktop — a triage tool:

- **Home**: plant cards (donut sliver + tier counts + open findings),
  then asset cards — tier chip, name, one sparkline (vel RMS 30 d), last
  measurement age, open-finding count. Swipe right: ack. Tap: asset
  sheet.
- **Asset sheet**: health timeline strip, three trend sparklines, the
  events/findings feed, and a "view spectra on desktop" hand-off (QR /
  link) rather than cramming waterfalls onto 390 px.
- Cards sort by (tier desc, direction desc, staleness); a global filter
  chip row (Critical / Alert / Monitor / stale sensors) — the whole app
  answers "what do I look at first" in one screen.
- Implementation: same token system, CSS container queries, bottom tab
  bar (Fleet / Inbox / Search); PWA manifest so it installs.

## 8. Desktop analysis suite

The explorer grows from viewer to workbench:

- **Cursors**: harmonic cursor sets (place at f, harmonics fan out),
  sideband cursors (carrier ± k·Δ), locked to order or Hz; delta
  readouts.
- **Kinematic overlays**: component kinematics (bearing catalog, gear
  teeth, vanes — schema already exists in v1 ontology) render as named
  forcing-frequency pins on every spectrum: BPFO/BPFI/BSF/FTF/GMF ± 
  sidebands, at the current speed estimate.
- **Waveform views** at truth points: time waveform, envelope,
  kurtosis/crest annotations, zoom-synced with the spectrum.
- **Cross-sensor compare**: two sensors side-by-side, synced time
  cursors (DE vs NDE is the first question every analyst asks).
- **Finding composer** docked right: select evidence → write → publish
  to the feed (§4).
- Everything already built stays: 2-D/3-D waterfalls, z view, Hz/order,
  pattern matrix, gate lane, byte ledger.

## 9. Reporting on demand

`POST /v1/reports {scope: plant|asset|sensor, period, sections[]}` →
server-rendered HTML (print-CSS = the PDF), stored and linkable:

- Executive: health distribution + tier changes + top findings + fleet
  stats (bytes, snapshots, coverage).
- Asset detail: health timeline, trends with ISO zones, pattern matrix
  extract, findings log with evidence thumbnails, snapshot/confidence
  summary (§6) — the compression audit belongs in the customer report.
- Layouts from the token system; charts rendered by the same canvas code
  (headless) so reports match the screen exactly. Scheduled reports =
  the same endpoint on a cron, emailed as link (phase 2).

## 10. Information architecture

```
Fleet (desktop)                    Mobile
├─ header: tier filter, search     ├─ Fleet (cards)
├─ left rail: plant/asset tree     ├─ Inbox (findings feed)
├─ health donut + KPI row          └─ Search
├─ asset card grid                 
└─ insight feed (right)            Asset page (both)
                                   ├─ health timeline + tier chip
Asset (desktop)                    ├─ trends (vel/acc/env, ISO zones)
├─ health timeline + events        ├─ pattern health matrix
├─ trends + snapshot lane          ├─ events & findings feed
├─ pattern health matrix           └─ desktop: analysis suite tabs
└─ analysis suite (explorer tabs)      (waterfall / 3-D / inspector)
```

## 11. Schema & API deltas

- `finding`, `finding_comment` tables (+ `event` rows on every change).
- `pattern_baseline` (sensor, rail, track_id, n, med, mad, window_start);
  `pattern.z REAL`.
- `health` columns on acquisition (tier SMALLINT, drivers JSONB) +
  `sensor_latest` extended; plant/asset rollup view.
- `sensor.params JSONB` (storage overrides, cutoff_ts), workspace params
  UI. `acquisition.pinned BOOL`.
- New endpoints: findings CRUD, reset-baseline, cutoff, audit,
  confidence, reports, health rollup. All scoped by workspace, all
  events, all ETag-cached like existing reads.

## 12. Phasing

1. **Health + pattern z** (server): pattern_baseline, health tiers,
   rollup endpoints — the model everything else renders.
2. **Fleet & asset pages** (desktop) + findings/events feed.
3. **Baseline reset / cutoff + snapshot lane + audit endpoints** — the
   confidence story.
4. **Mobile PWA** (cards, inbox, asset sheet).
5. **Analysis-suite upgrades** (cursors, kinematic pins, waveform views,
   compare) and **reports**.

Anti-goals, stated: no automatic fault *diagnosis* labels without a human
finding (v1 removed exactly that feature for over-claiming); no per-user
permissions inside a workspace (that's the passphrase model's honest
boundary — if it hurts, the answer is multiple workspaces, not pretend
roles).
