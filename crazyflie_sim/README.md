# Simulator Logging and Flight Recording

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

`auto_client` runs a fixed, simulator-only acquisition sequence. Start the
updated simulator with `--record` first, stop the Xbox client and other command
senders, and leave the vehicle stationary on the ground:

```bash
python crazyflie_sim/auto_client.py --host localhost --port 8000
```

The client checks the simulator identity and recorder status before sending any
commands, including sufficient hover-thrust margin for the bounded profile.
Each trial resets the grounded simulator, ramps up to a 1 m hover,
waits for settling, excites one axis at a time, then lands and sends zero thrust.
Zero thrust and motor outputs are checked before file retrieval begins.
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

Defaults are one trial, 32 seconds of stationary excitation per phase,
100 Hz commands scheduled against simulation time, and acceptance over
0.5-15 Hz. Commands are seeded, bounded broadband binary signals with ramped
phase boundaries. Their default amplitudes are 2 deg for Roll/Pitch, 5 deg for
Yaw angle and 10 deg/s for Yaw rate. Simulator-side logging still records every
control step; client HTTP polling is not used as the analysis data source.

```bash
python crazyflie_sim/auto_client.py --repeats 3 --duration 40 \
  --command-hz 100 --seed 7 --log-dir crazyflie_sim/logs
```

Use `--help` for bounded amplitude and frequency options. Commands faster than
the simulator control loop are rejected. Missed updates are skipped, never
burst-sent or assigned invented timestamps. Clock stalls, session changes,
network failures and position/attitude/rate limit violations abort acquisition
and trigger a best-effort simulator zero-thrust stop. An abort may drop the
simulated vehicle; this client must not be adapted to hardware without a
separate hardware safety design.

Default safety bounds are 15 deg tilt, 120 deg/s body rates, 2 m/s speed, a
1.5 m horizontal radius and height no more than 0.7 m above the target. After
takeoff, height below 0.25 m also aborts the run. Ctrl-C and SIGTERM run the stop
cleanup; a hard process kill cannot guarantee delivery of a stop command.

Completed source CSVs are fetched through the read-only `/recording/status`
and `/recording/file?name=...` endpoints. Only files completed in the current
recorder session can be downloaded. File transfer and analysis take place after
landing. The client preserves the original downloaded CSVs and crops only the
stationary excitation intervals into separate trial/phase files, without
resampling, scaling or shifting timestamps.

Results are stored in a unique `logs/auto_<session>/` directory:

```text
raw/                         # Unmodified downloaded recordings
trial_001_roll.csv            # Cropped analysis inputs
trial_001_pitch.csv
trial_001_yaw_angle.csv
trial_001_yaw_rate.csv
reports/                     # response_analysis reports
quality.json                 # Measured acceptance results
```

Select the same phase files when constructing sim/real comparison groups; do
not indiscriminately combine all phases as independent repetitions for every
channel.

### Quality Gate

A completed flight is not automatically a successful acquisition. The client
runs `response_analysis` on the actual saved CSVs using a 4 s window, 2 s
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
- No substantial, strongly correlated other-axis input or recorded
  attitude/rate safety-bound violation is found.

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
