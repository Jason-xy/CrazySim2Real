# Simulator Logging and Flight Recording

## Runtime and Coordinates

The simulator targets **Isaac Lab v3.0.0-beta2.patch1**
(`ffff603eafc6b74264a5261cc0183d6a65390d78`) with **Isaac Sim 6.0.1** and
the PhysX backend. Isaac Lab 2.x is not supported. The launch script reuses
`crazyrl-isaaclab:3.0.0-beta2.patch1`, shared with CrazyE2E / crazy_rl-ctbr,
and builds it only if missing. The Dockerfile matches that project's RL runtime,
including RSL-RL 5.0.1, pynput, debugpy, and rtree; PyTorch is selected by the
pinned Lab installer.

All project quaternion arrays use **XYZW**, with identity `(0, 0, 0, 1)`.
The orientation maps body coordinates to world coordinates. Initial poses,
resets, state reads, rotation math, and marker orientations share this order.
There is no quaternion-order detection or legacy compatibility mode.
Controller/API/ROS attitude values remain Euler angles in degrees, and the
analysis CSV remains Euler angles and FLU body rates, not quaternion components.

The default is a GUI (`--viz kit`). For automatic acquisition, pass `--viz none`;
`--gui` and `--headless` are wrapper aliases for `kit` and `none`, respectively.
Conflicting visualization flags are rejected. Cameras, streaming and optional
Kit extensions are not enabled implicitly. GUI launch requires `DISPLAY`;
headless launch does not grant X server access. Only X authorization added by
the launch script is revoked on exit.

GUI drawing is decoupled from physics, control and recording. The default
`--render-interval 4` draws once per four physics steps (50 frames per simulated
second at the default 200 Hz control rate). Increase it to 8 if drawing remains
too expensive; the control and CSV sample rates do not change. The initial
camera frames the flight area, and robot visibility uses the SDK's standard
visibility API rather than a nonstandard USD token.

Application logs and controller recordings are separate:

```text
crazyflie_sim/logs/
  simulator.log
  xbox_client.log
  auto_client.log
  flight_<session>_<pid>_001_attitude.csv
  flight_<session>_<pid>_002_position.csv
```

The default directory is anchored to the package location, not the working
directory. The simulator and both clients accept `--log-dir PATH`. Text logs append to
existing files; no history is automatically pruned.

On Xbox client startup, an old `xbox_client.log` beside the script or in the
working directory is moved into the selected log directory. If a destination
already exists, the old file is preserved as a uniquely named history file.
The original is only removed after a successful copy. Imports do not create
logs or start recording.

## Record

In the existing Docker/Isaac Lab environment, add `--record` to the simulator
launch command:

```bash
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/run.py \
  --port 8000 --record
```

For unattended recording without a GUI:

```bash
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/run.py \
  --host localhost --port 8000 --record --viz none
```

Or pass `--record --log-dir /path/to/logs` to `crazyflie_sim/run.py` when using
your existing Isaac Lab launcher. Ensure the directory is writable by the
process; shared host/container directories need appropriate permissions.

The Xbox client can independently select its text log directory:

```bash
python crazyflie_sim/xbox_client.py --host localhost --log-dir crazyflie_sim/logs
```

Recording is off by default. When enabled, it records from startup until exit,
including ground, takeoff and landing data. It does not detect airborne state
or change PID gains, control frequencies, commands or control modes.

## Samples and Segments

Each controller update contributes one row, even when a reference is held
constant. Sampling follows the actual simulation step, not the Xbox/network
update rate.

Samples are captured after control computation and before physics advances.
The timestamp comes from the same state snapshot used for that control update.
The simulator converts the controller's internal references and feedback into
independent Python numbers before submitting them to the writer.

The CSV has the existing 13 analysis columns: `t_s` plus angle and body-rate
reference/measurement pairs for three axes. Units are seconds, degrees and
degrees/second.

| Controller mode | Angle columns | Body-rate columns |
| --- | --- | --- |
| ATTITUDE | Roll/Pitch populated; both Yaw fields empty | All three axes |
| POSITION / VELOCITY | All three axes | All three axes |
| ATTITUDE_RATE | All angle fields empty | All three axes |

The table describes recording behavior, not new control interfaces; this
change does not add an Xbox or HTTP rate-control mode.

Mode changes, resets and backwards simulation time start new files. Each file
starts at `t_s=0`; equal timestamps within a segment are not written twice.
Step and reset operations are serialized, and reset requests still return only
after the reset has completed.

## Writing and Shutdown

One background thread writes CSV data through a 2048-row queue and flushes at
least every 0.5 seconds when the filesystem is responsive. The control loop
does not wait for the CSV writer.

Files being written end in `.csv.partial`. A healthy segment is published as
`.csv` after flushing and closing. Publication uses an exclusive same-directory
hard link so an existing CSV cannot be overwritten. A filesystem that does not
support that operation causes a recording error and leaves the partial file.

Queue overflow, snapshot failure or write failure disables further recording
and reports an error. Completed older segments are kept; unfinished data keeps
its `.partial` suffix. Accepted rows are drained on normal close; a disk error
can prevent pending rows from being saved. These failures do not issue flight
commands or stop the control loop.

Graceful exit stops the API, drains/closes recording, then closes the simulation
application. An exception during the run leaves the active recording partial.
Do not treat a partial file as a completed experiment.

## Analyze

Select a completed CSV and, if needed, a flight interval relative to that file:

```bash
python -m response_analysis validate /path/to/flight.csv
python -m response_analysis analyze /path/to/flight.csv \
  --interval 5 25 --out /path/to/new-report
```

Use separate files or analysis intervals for the maneuvers you intend to
compare. Recording segments from one flight are not automatically independent
repetitions for statistical analysis.

## Automatic Acquisition

`auto_client` runs simulator-only identification and trajectory acquisition. Start the
updated simulator with `--record` first, stop the Xbox client and other command
senders, and leave the vehicle stationary on the ground:

```bash
python crazyflie_sim/auto_client.py --host localhost --port 8000 --mode both
```

The client checks the simulator identity and recorder status before sending any
commands, including nominal trajectory feasibility with the actual total vehicle
mass (body plus propellers). `--mode` selects `identification`, `agile`, or `both`
(default). Each combined trial resets the grounded simulator, rises to 1 m,
performs four identification phases, climbs to 4 m, flies four trajectories,
then lands and sends zero thrust. Ground contact and stopped motor outputs are
checked before file retrieval begins. A default combined trial takes approximately
6-7 simulation minutes without recovery interventions.
It does not start Isaac Lab, connect to real hardware, or change firmware/PID
parameters. Run only one control client at a time.

| Phase | Existing control interface | Quality-checked channels |
| --- | --- | --- |
| Roll angle excitation | ATTITUDE | `angle.roll`, `rate.x` |
| Pitch angle excitation | ATTITUDE | `angle.pitch`, `rate.y` |
| Yaw angle excitation | POSITION | `angle.yaw` |
| Yaw rate excitation | ATTITUDE | `rate.z` |

Roll/Pitch rate data comes from the internal rate references generated by the
angle controller, not a new direct CTBR interface. Other axes are held near
hover using small bounded position/heading corrections.
Returns to hover use quintic transitions matching reference position, velocity
and acceleration before handing back to POSITION control for the settling check.

Identification defaults are 32 seconds of stationary excitation per phase,
100 Hz commands scheduled against simulation time, and acceptance over
0.5-15 Hz. Commands are seeded, bounded broadband binary signals with ramped
phase boundaries. Their default amplitudes are 2 deg for Roll/Pitch, 5 deg for
Yaw angle and 10 deg/s for Yaw rate. Simulator-side logging still records every
control step; client HTTP polling is not used as the analysis data source.

### Agile Trajectories

All agile trajectories use the existing ATTITUDE interface, not CTBR. A client
tracker combines analytical acceleration feedforward with position/velocity
feedback (`Kp=(2,2,4)`, `Kd=(2.8,2.8,3)`) and measured-tilt thrust compensation.
The final API limits remain +/-30 deg roll/pitch, +/-120 deg/s yaw rate and
normalized thrust in `[0,1]`; firmware gains and motor saturation are unchanged.

| Trajectory | Default shape | Three phase speeds (rad/s, not body rates) |
| --- | --- | --- |
| Circle | 2 m radius, fixed height | 0.8 / 1.2 / 1.5 |
| Figure eight | `x=2 sin(theta)`, `y=sin(2 theta)` | 0.55 / 0.8 / 1.0 |
| Helix | 2 m radius, `z=4+0.75 sin(theta/2)` | 0.8 / 1.2 / 1.5 |
| Shuttle | +/-2 m along a horizontal diagonal | 0.6 / 0.9 / 1.2 |

Each path lasts 48 simulation seconds, divided into three speed stages with
four-second quintic start, speed-change and stop ramps. The shuttle adds 20%
phase-speed modulation with an eight-second period and a ramped +/-60 deg,
0.25 Hz heading oscillation. Other paths hold the initial heading. A seed fixes
path rotation and direction; these parameters are recorded in `manifest.json`.
Default nominal paths stay at or above 3.25 m. Preflight also checks their
returns to hover against clearance, attitude, yaw-rate and thrust capability.
This kinematic check is not proof of closed-loop safety.

```bash
python crazyflie_sim/auto_client.py --mode identification --duration 40
python crazyflie_sim/auto_client.py --mode agile --trajectory-duration 48 --agile-height 4
python crazyflie_sim/auto_client.py --mode both --repeats 3 --seed 7
```

`--duration` affects identification only; `--trajectory-duration` affects each
agile path (minimum 24 seconds). `--height` remains the identification/takeoff
height; `--agile-height` selects the agile reference altitude.

### Ground Recovery and Failure

Phases use simulation time. `--phase-timeout 900` independently limits how many
wall-clock seconds a phase or hover check may take (default: 15 minutes).
A slowly advancing GUI is not a stalled clock: the five-second clock-stall
watchdog, network timeouts and telemetry validation remain active. Keep the
GUI timeline playing during automatic flight. When a wall timeout occurs, the
error reports both completed and requested simulation seconds.

Use `--help` for bounded amplitude and frequency options. Commands faster than
the simulator control loop are rejected. Missed updates are skipped, never
burst-sent or assigned invented timestamps. Clock stalls, session changes,
and network failures interrupt acquisition. The old 15 deg tilt, 360 deg/s body
rate, 2 m/s speed, 1.5 m radius, upper-height and `[0.08,0.65]` thrust gates have
been removed from the automatic client and its saved-data safety checks.

The client predicts clearance using downward velocity, a 0.5 s response allowance,
and braking distance with at most 3 m/s2 upward deceleration, further reduced
by the current model's available upward thrust. Predicted clearance below 0.5 m
triggers leveling and climbing before a smooth return to the interrupted path.
Only path progress pauses: physics and recorded timestamps continue unchanged.
Corrections are labeled and excluded from nominal trajectory CSVs; the complete
raw recordings retain them. A disrupted identification phase fails rather than
being presented as stationary excitation.

The simulator checks contact forces each physics step and latches contact events
in `/state`. Initial ground contact and controlled low-speed landing are distinct
from unexpected airborne contact. An unexpected contact immediately ends the
entire acquisition and resets the simulator; it does not retry or continue with
another trajectory. Failed segment data and the pre-reset event are retained.
Normal interruption, Ctrl-C and SIGTERM attempt controlled landing before zero
thrust; they do not intentionally cut power to an unconfirmed airborne vehicle.
Communication loss, severe loss of control, or a hard process kill can still
prevent recovery. This behavior is simulation-only, not a hardware safety design.

Completed source CSVs are fetched through the read-only `/recording/status`
and `/recording/file?name=...` endpoints. Only files completed in the current
recorder session can be downloaded. File transfer and analysis take place after
landing or collision reset. `POST /recording/split` seals a segment without
changing flight mode or waiting for the next sample. The client preserves the
original downloaded CSVs and crops the identification or nominal trajectory
intervals into separate trial/phase files, without
resampling, scaling or shifting timestamps.

Results are stored in a unique `logs/auto_<session>/` directory:

```text
raw/                         # Unmodified downloaded recordings
trial_001_roll.csv            # Cropped analysis inputs
trial_001_pitch.csv
trial_001_yaw_angle.csv
trial_001_yaw_rate.csv
trial_001_agile_circle.csv
trial_001_agile_figure8.csv
trial_001_agile_helix.csv
trial_001_agile_shuttle.csv
reports/                     # Identification response_analysis reports
quality.json                 # Completion, identification and agile coverage
manifest.json                # References, intervals, corrections and contact events
```

The CSV schema still contains only the six paired angle/rate channels. ATTITUDE
mode leaves both yaw-angle columns empty: a heading goal is not an active
yaw-angle controller reference. No position, velocity, quaternion or motor
columns are added.

Select the same phase files when constructing sim/real comparison groups; do
not indiscriminately combine all phases as independent repetitions for every
channel.

### Quality Gate

A completed flight is not automatically a successful acquisition. The client
runs `response_analysis` on identification CSVs using a 4 s window, 2 s
response horizon, at least eight excited windows and coherence threshold 0.8.
Each target must satisfy all of the following:

- The recording covers the requested excitation interval, has no long gaps,
  and its effective sampling rate is at least 95% of the simulator rate.
- At least 90% of the requested command updates were sent, without gaps longer
  than three command periods.
- Reference standard deviation exceeds 0.1 deg for angles or 1 deg/s for rates.
- At least 80% of the requested frequency-band bins pass the analyzer's
  coherence and excitation-power gates.
- DC gain, rise time and settling time are resolved. A peak time is not required
  for monotonic responses.
- No substantial, strongly correlated other-axis input is found.

Agile phases require complete nominal intervals, valid paired channels and the
same recording/command cadence checks, but do not require broadband coherence.
Coverage is evaluated over the combined saved agile samples: the P95 absolute
measured roll and pitch must each reach 15 deg, and P95 absolute measured yaw
rate must reach 60 deg/s. Large references alone cannot pass. Reports include
RMS, peaks and P95 of the measured angle/rate channels. Incomplete acquisition,
collisions and insufficient coverage cannot produce an overall pass.

Exit code `0` means every target passed, `3` means flight completed but measured
quality failed, and `2` means acquisition/configuration failed. Failures retain
data and reports; the client does not loosen thresholds, increase amplitudes or
retry indefinitely to manufacture a pass. Insufficient timing resolution or a
slow response outside the configured horizon may still fail even for a valid
CSV. Inspect the report rather than treating the result as a guarantee of
unbiased physical-system identification.

## Offline Checks

```bash
python -m unittest discover -s crazyflie_sim/tests -t . -v
python -m unittest discover -s response_analysis/tests -t . -v
```

Tests cover recording, failure handling, path/history handling, and the actual
manager step/reset hooks using fake simulator objects. Auto-client tests use an
HTTP-shaped numerical test double and actual CSV recording/analysis, including
deliberately bad feedback. They do not start Isaac Lab or connect to hardware.
Actual flight stability, GPU copy overhead, network cadence and simulator
throughput still need measurement in the intended runtime.

For explicit real-SDK acceptance, start a dedicated updated simulator with
`--record` on an unused port, then run:

```bash
python crazyflie_sim/tests/auto_acceptance.py --port 18009 --seeds 0 1 2 \
  --log-dir crazyflie_sim/logs/acceptance_new
```

The output directory must not already exist. This runs real simulator flights
and checks completion, measured coverage, absence of recovery/contact events,
and observed agile height/speed. The separate `--fault-only` option deliberately
cuts simulated thrust to verify collision termination, reset and partial-data
preservation. Never run that option against another user's flight.

## Explicit SDK Check

This check starts real Isaac Lab. It is separate from offline test discovery.
Use a new output directory each time:

```bash
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/tests/isaaclab_smoke.py \
  --viz none --log-dir /workspace/isaaclab/CrazySim2Real/crazyflie_sim/logs/sdk-check
```

It verifies XYZW rotations, reset state, effective body mass/inertia,
non-accumulating body-frame wrenches, physics-step timing and completed CSVs.
Repeat with `--viz kit` and another output directory to exercise the GUI.
Passing this ground check does not mean automatic flight or response quality passed.
