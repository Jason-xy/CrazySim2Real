# Repository Guidance

## Project Overview

CrazySim2Real benchmarks Crazyflie control algorithms in simulation and on real
hardware. This file provides repository-wide guidance; see `README.md` for usage.

- `crazyflie_benchmark/main.py`: benchmark CLI and experiment orchestration.
- `crazyflie_benchmark/core/`: configuration, connections, logging, metrics, and safety.
- `crazyflie_benchmark/controllers/`: shared controller interface and sim/real adapters.
- `crazyflie_benchmark/tests/` and `crazyflie_benchmark/test_plans/`:
  flight experiments and YAML plans.
- `crazyflie_benchmark/tools/`: log analysis and hover-thrust measurement utilities.
- `crazyflie_sim/`: Isaac Lab simulator, HTTP API, and teleoperation/visualization clients.
- `scripts/` and `docker/isaaclab/`: setup scripts and container configuration.
- `docker/isaaclab/IsaacLab` and `crazyflie_sim/controllers/cf_controller`: Git submodules.

## Development Commands

Run all commands below from the repository root. Setup and runtime commands are
not routine validation steps.

### Environment Setup

Install benchmark dependencies in the intended Python environment:

```bash
python -m pip install -r crazyflie_benchmark/requirements.txt
```

For requested first-time simulator setup, initialize the submodules:

```bash
./scripts/init.sh
```

This script runs a recursive submodule update. Inspect existing submodule changes
first; do not run it automatically or overwrite a user's checked-out revisions.

### Offline Checks

These commands show CLI options and experiment types without connecting to a drone:

```bash
python crazyflie_benchmark/main.py --help
python crazyflie_benchmark/main.py --list-tests
git diff --check
```

### Simulation Operations

Simulation requires Docker Compose, an NVIDIA GPU, and the Isaac Lab environment.
The following command starts a container; its script also changes X display access
with `xhost +`. Do not use it as an offline smoke test.

```bash
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/run.py --port 8000
```

With the simulator running, use a separate terminal to run a simulation benchmark:

```bash
python crazyflie_benchmark/main.py \
  --config crazyflie_benchmark/config/simulator_config.yaml \
  --plan crazyflie_benchmark/test_plans/step_tests.yaml
```

Before running, verify the config exists, loads successfully, and selects
`connection_type: simulator` with the intended host and port. Missing or invalid
configuration can fall back to real-hardware defaults. The `--sim` flag alone
does not select the simulator backend in the current implementation.

## Engineering Constraints

- Match the user's language in replies; keep this document and code identifiers in English.
- Read the relevant implementation before changing it. Resolve routine details from
  the repository; ask only when ambiguity materially affects correctness or safety.
- Keep changes focused on the requested outcome. Preserve existing behavior unless
  the task explicitly calls for a change, and avoid unrelated cleanup or new dependencies.
- Preserve user changes, including submodule revisions. Do not automatically update
  submodules, delete logs/caches, or stop all simulator containers.
- Keep simulation and hardware behavior compatible at the shared controller interface.
  Benchmark setpoints use degrees, degrees/second, and raw thrust in `0-65535`.
  The simulator adapter normalizes attitude commands and thrust for the HTTP API;
  do not apply those normalized units to the shared benchmark interface.
- Preserve safety clamping, zero-thrust stop behavior, disarming, and shutdown cleanup.
  Without explicit user authorization, do not connect to real hardware, arm, take off,
  send flight commands, or weaken safety limits.
- Keep experiment-duration clocks aligned with simulator/vehicle telemetry where
  configured. Do not replace them with wall time; control-loop pacing uses monotonic time.

## Validation and Delivery

- Define success from the requested behavior and constraints, not a fixed ritual.
  Use the smallest relevant checks, expanding coverage when the change warrants it.
- `crazyflie_benchmark/tests/` contains flight experiment implementations, not a
  conventional automated unit-test suite. Do not equate running a flight plan with
  running offline tests or claim an established repository-wide test/lint command.
- For documentation-only work, verify paths and CLI examples, inspect the diff,
  and run `git diff --check`. Do not install dependencies or start simulation for it.
- For code changes, check the affected behavior using available tests or focused
  offline checks. Use mocks for hardware boundaries unless real operation is authorized.
- Before finishing, check that only intended files changed. Report the outcome,
  checks actually run, and relevant failures, skipped checks, or environment limitations.
  Do not claim simulation or hardware validation from CLI help or syntax checks.

## References

- [OpenAI: Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md/).
- [OpenAI: GPT-6 Astra guide, Prompting Best Practices](https://developers.openai.com/api/docs/guides/latest-model).
  Consulted on 2026-09-08 for clear outcomes, constraints, completion criteria, and
  proportionate verification. This rolling guide is a reference, not a model requirement.
- Eric Provencher (@pvncher), [Rethinking skills and prompts for GPT-6 Astra](https://x.com/pvncher/article/2095991462416490862):
  further reading only. As of 2026-09-08, only the title and summary were verified;
  the X article body was inaccessible and its full text has not been verified.
