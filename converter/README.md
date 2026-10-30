# spine-il-converter

Converts recorder **schema v2** sessions (tracked endoscope + drill demonstrations over a virtual
anatomy) into a **LeRobot v3** dataset for ACT / GR00T N1.7 training.

Two input formats, told apart per file (`load_session` does it):

- **Recorder v2 recordings** (default) — what the viewer's tracker recorder actually writes
  (`rec-2.x`, `sessionRecordSchemaV2*.ts`): `hdr` → `f` / `ev` / `cut` → `ftr`, no viewer captures yet.
  See [Recorder v2 recordings](#recorder-v2-recordings).
- **PLAN-0 fixtures** — the synthetic §5.3 schema with `m` / `img` rows and viewer captures
  (`make-fixtures`, `recorderVersion` ending in `-fixture`). See [PLAN-0 fixtures](#plan-0-fixtures).

```
DATA_ROOT/
  sessions/<YYYY-MM-DD>/<sessionId>.jsonl     recorder v2: hdr → f / ev / cut → ftr  (PLAN-0: + m / img)
  frames/<sessionId>/<cam>/<frameIdx>.jpg     PLAN-0 only: cam ∈ {endoscope, scene}, 256×256
```

## Usage

```bash
pip install -e ".[dev]"                      # needs lerobot[dataset] 0.6.x, Python ≥ 3.12

spine-il-convert validate DATA_ROOT/sessions/2026-09-20/*.jsonl   # contract + filter check per file
spine-il-convert build --data-root DATA_ROOT --out datasets/spine-lam-lt45-v1 \
    --repo-id <hf_user>/spine-lam-lt45-v1 [--overwrite] [--replay-verdicts replay_verdicts.parquet]
spine-il-convert make-fixtures /tmp/fixtures  # synthetic sessions: 3 includable + 8 rejectable

pytest                                       # unit + end-to-end (builds a dataset, trains ACT for 2 steps)
pytest -m "not slow"                         # unit tests only (~2 s)
SPINE_REC_SAMPLE=/path/to/s_…_trk_0551.jsonl pytest tests/test_recorder_v2_sample.py  # a real recording (not committed)
```

Recorder v2 flags: `--allow-core` (accept `timingSpec` `v2-core` files), `--allow-robot-tools` (accept files whose
tracker also ran robot-mounted marker blocks), `--max-masked-frac` (default 0.05), `--split-on-gap` (off by default),
`--task TEXT` (LeRobot task text).

## Recorder v2 recordings

| key | shape | content |
|---|---|---|
| `observation.state` | float32[19] | endo tip(3) + rot6d(6), drill tip(3) + rot6d(6), `drill_trig` — basal frame, mm |
| `action` | float32[19] | absolute state of the next frame (valid when the next frame's `mask` is) |
| `observation.endo_roll_deg` | float32[1] | the viewer's endoscope roll (deg) for the pose at this frame, unwrapped in the episode |
| `observation.tracker_time` | float32[1] | seconds since the episode's first frame (camera-frame axis) |
| `label.drill_cut` | float32[1] | 1 when the drill engine stamped with a drill pose from a camera frame in (t_{k−1}, t_k] |
| `mask` | float32[3] | `endo`, `drill` pose valid · `drill_trig_known` |
| `diag.tsrc_minus_texp_ms` | float32[2] | tracker send time − camera frame per tool (diagnostic) |

`label.*`, `mask` and `diag.*` are not policy inputs: LeRobot maps only `observation.*` / `action*` keys to policy
features.

- **Row order** is `r`, not the file's line order: rows are sorted by `r`, a repeated `r` (re-sent chunk) keeps the
  first copy, gaps in `r` (lost chunks) are counted, `hdr` may sit anywhere. Unknown row kinds and keys are ignored,
  and `recorderVersion` is only kept as metadata — the format is `schemaVersion` 2 + `timingSpec`. A malformed row
  (bad quaternion or shaft, wrong types) is masked like any invalid row instead of stopping the build; a registration
  that is not a rotation, or a `sessionId` outside `[A-Za-z0-9_.-]{1,120}` (it names sidecar folders), is a schema
  error.
- **Time axis** = `tExp`, the camera-frame stamp (epoch ms) that both tools of one frame share. Frames sit on a 30 fps
  grid anchored on the first camera frame where both tools are valid; an episode's frame 0 is the first grid frame
  where both are valid. `tSrc` (tracker send, ≈ +24.5 ms) only feeds the `diag` column.
- **Poses**: tip = `scene.tip` (the viewer's composeToolPose, LPS mm). Tool frame: z = `scene.u` (shaft, toward the
  tip), x = the marker-plane normal (tool local +z) of the full rotation T.R · R(`quat`) projected ⟂ z, y = z × x —
  so the twist about the shaft is kept. `up` is not used: the relay stamps every pose with one constant calibration
  vector, which would erase the roll. A dynamic registration uses each row's `T`. The x axis follows the tool model's
  marker frame, so episode stats record each tool's model (`identity.tool_models`) and the dataset report warns when
  one dataset mixes models.
- **Endoscope roll** follows the viewer (useTrackerFilmDrive → trackerEndoMap): marker-plane normal flipped to face
  the camera (sign kept while |cos| < 0.1), ⟂ shaft, measured about the shaft from scene +Z ⟂ shaft. The viewer's own
  value for a message (`real.roll`, or `drawn.roll` of a row whose screen showed that message — joined by
  `seq`/`tSrc`) is used when recorded, the replicated rule otherwise; `endo_roll_rule_err_max_deg` reports their
  largest disagreement. The endoscope's local +z has faced the camera in every rig session so far, so its x axis
  is the viewer's camera up and rot6d carries the same roll.
- **Validity**: a row is a valid pose unless `lost`, `hold` (jump-hold re-send), without `scene`/`T`, malformed, or
  `vis` < 4; `still` rows are valid. A frame is valid when it is a valid row's frame or lies between two consecutive
  valid rows ≤ `--max-gap-ms` apart. Other frames hold the last valid sample with mask 0 — invalid rows are never
  bridged by interpolation. Episodes are trimmed to the first/last frame with both tools valid, then excluded when
  a tool is masked in more than `--max-masked-frac` of the episode's frames (`masked`), or when a masked run spans
  more than `--max-gap-ms` (`gap`). Rows lost across the whole recording only warn (above `--max-lost-frac`).
- **`--split-on-gap`** (off by default): a masked run longer than `--max-gap-ms` cuts the recording into pieces instead
  of excluding it — never bridged. Each piece is filtered on its own (≥ 3 s, masked fraction) and becomes its own
  LeRobot episode; the pieces of one recording share its split (`piece` in episodes.jsonl, dropped pieces listed).
- **`drill_trig`** = `btn` bit TRIG (ZR held) of the latest drill row at the frame (zero-order hold). Rows without
  `btn` (state unknown) keep the last known value and set `drill_trig_known` = 0. This is what an arm commands.
- **`label.drill_cut`** = 1 when a stamp — bone or soft tissue — was made with a drill pose from a camera frame in
  (t_{k−1}, t_k]: `cut` rows (`cut.tExp`) when the recorder writes them (`hdr.replay.cutRows`, rec ≥ 2.3), else drill
  rows with `proxy.stamped` = 1 placed at their `drawn.tExp`. It is a label on pose time: the engine applied the stamp
  later (0551: p50 ~120 ms, up to 0.58 s), so at frame k a policy could not have observed it. Cut rows without a
  time on the axis are reported, not placed. Stamps count whatever voxel change they made (`mod` may be 0).
- **Basal frame**: origin = the endoscope's skin portal from the viewer's entry points (`drawn.entry`, else
  `real.entry`) while the scope is inserted (depth > 0). Each distinct recorded entry value counts once, so a screen
  that did not redraw or an entry re-used after a failed skin cast adds no evidence. Rule: the earliest 1 s window with
  ≥ 8 distinct values all within 2 mm of their median (`inserted_stable_window`); else the median of all inserted
  values (`inserted_median`); else `hdr.basal`; else excluded. z = the median view direction of the same values,
  x = cranial ⟂ z, y = z × x. Any rule but `inserted_stable_window` warns, and so does a portal more than 10 mm from
  the median of all inserted entries (the scope was inserted at more than one place). z comes from the first stable
  second, so the frame's direction differs a little between recordings.
- **Gates**: rejected unless allowed — `timingSpec` `v2-core` (`timing_core`), robot-mounted marker blocks
  (`robot_tools`: `endo2` in the tracker's tools, or a robot drill model tracked as `drill` — a calib file named
  `*robot*` or a known robot model sha). Warned — `baseline` null, `baselineOverride` / `t0Override`,
  `toolCalib.<t>.sha256_12` ≠ `tracker.cfg.tools.<t>.calib.sha12`, `replay.carvedBeforeStart`, a marker normal
  within 10° of the shaft, stamps or ZR presses outside the converted frames. Also excluded: no footer,
  `operatorVerdict` `bad`, a tool without rows (`no_rows`) or valid poses, non-finite values, too short.
- **Sidecars** (`meta/spine_il/`): `episodes.jsonl` (source, split, piece, `t0_ms`, stats incl. basal, roll, drill,
  warnings and identity hashes), `conversion_report.json` (incl. dataset-level warnings and exclusions such as a
  second file with an already converted `sessionId`), and per recording `recordings/<sessionId>/hdr.json` +
  `cut.jsonl` — the `hdr` row and every `cut` row verbatim, in `r` order (full-precision stamp centres; re-applying
  them in order reproduces the carve). A cut row's frame is `ceil((cut.tExp − t0_ms) / (1000 / fps))`.

One recording = one episode (or its pieces); no image streams until the recorder writes viewer captures.
GR00T: `--policy.use_relative_actions=true --policy.relative_exclude_joints='["drill_trig"]'`
(`drill_trig` is the only action name containing that substring).

**ACT needs images or `observation.environment_state`** (LeRobot 0.6.1). The converter writes neither yet; that is a
training-design decision. If an image-free baseline needs one, use context a deployed policy can also observe, aligned
to when the engine applied it (`tMono` → epoch), e.g. removed volume so far (Σ cut `mod` × voxel volume),
`proxy.touching`, `proxy.depthMm`. Do not use `label.drill_cut` (future relative to the frame and the frame's own
outcome), `observation.tracker_time` (leaks episode progress) or `observation.endo_roll_deg` (already in the endoscope
rot6d).

## PLAN-0 fixtures

| key | shape | content |
|---|---|---|
| `observation.state` | float32[19] | endo pos(3) + rot6d(6), drill pos(3) + rot6d(6), `drill_on` — basal frame, mm |
| `action` | float32[19] | absolute state of the next frame |
| `observation.images.endoscope`, `.scene` | video 256×256×3 | viewer captures |
| `observation.phase` | float32[5] | one-hot phase (footer = human-confirmed, else auto) |
| `observation.exposure` / `.progress` / `.goal` | float32[2] / [3] / [6] | lesion exposure, removal progress, lesion area in basal frame |
| `observation.tracker_time` | float32[1] | seconds since the episode's first frame |

Basal frame: origin = skin entry, z = entry→target, x = cranial (+S, LPS) projected ⟂ z, y = z × x.
rot6d = first two columns of the rotation matrix, column-major. Tool frame: z = shaft (toward tip),
x = `scene.up` orthogonalised.

Processing: both tool streams are resampled onto a 30 fps grid phase-locked to the endoscope captures
(linear position, slerp rotation); metrics and captures are matched by `tSrcEndo` within ±20 ms.
Dropouts ≤ 250 ms are bridged (captures during the dropout may be stale and are counted); longer ones
exclude the episode. Episodes are excluded for: schema errors, no footer, verdict `bad`, fail flags,
replay-infeasible, tracker loss > 5 %, no laminectomy phase, basal-frame parity mismatch, gaps, too
short. Splits (train/val/test = 80/10/10, test = held-out basal vectors) are stable under appends.

Provenance per episode (source file, absolute tracker t0, split, bridging stats) is written to
`meta/spine_il/episodes.jsonl`; exclusions and config to `meta/spine_il/conversion_report.json`.

GR00T: `--policy.use_relative_actions=true --policy.relative_exclude_joints='["drill_on"]'`
(`drill_on` is the only action name containing that substring). ACT has no relative-action option
in LeRobot 0.6.1 and trains on the absolute basal-frame actions.
