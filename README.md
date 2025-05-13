# CrazySim2Real: A Sim2Real Benchmarking Framework for Crazyflie Drones

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CrazySim2Real** is a comprehensive framework for evaluating control algorithms on Crazyflie quadcopters, providing a standardized testing environment that bridges the gap between simulation and real hardware. This repository contains two main components:

1. A physics-based Crazyflie drone simulator built on NVIDIA Isaac Lab
2. A benchmarking framework for evaluating controller performance in both simulation and real-world environments

<img src="https://www.bitcraze.io/images/crazyflie2-1brushless/brushless_585px.jpg" alt="Crazyflie Drone" width="300"/>

## Repository Structure

```
CrazySim2Real/
├── crazyflie_benchmark/     # Benchmarking framework for controller evaluation
│   ├── config/              # Configuration files for hardware and simulator
│   ├── logs/                # Test result logs (both sim and real)
│   ├── test_plans/          # Predefined test plans (step, impulse, sine sweep)
│   └── tests/               # Test implementation modules
├── crazyflie_sim/           # Simulator implementation
│   ├── api/                 # API server for simulator control
│   ├── controllers/         # Controller implementations for simulator
│   └── sim/                 # Simulation manager and physics implementation
├── docker/                  # Docker configurations
│   └── isaaclab/            # Isaac Lab simulator environment
└── scripts/                 # Utility scripts for setup and execution
```

## Key Features

- **Unified Testing Interface**: Run identical tests on both simulated drones and real hardware
- **Standardized Test Protocols**: Pre-defined step, impulse, and sine sweep test plans
- **Data Collection & Analysis**: Automated logging, metrics generation and visualization
- **Physics-Based Simulation**: High-fidelity drone simulation using NVIDIA Isaac Lab
- **Configurable Controllers**: Evaluate different control algorithms with minimal code changes
- **Safety Monitoring**: Built-in safety features to protect hardware during testing
- **Docker Integration**: Containerized simulation environment for consistent results

## Getting Started

### Prerequisites

- Python 3.8+ with pip
- Docker and Docker Compose (for simulation)
- Crazyflie hardware and Crazyradio PA (for real hardware tests)
- NVIDIA GPU with latest drivers (for simulation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CrazySim2Real.git
   cd CrazySim2Real
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r crazyflie_benchmark/requirements.txt
   ```

3. Set up the Isaac Lab Docker environment (only needed for simulation):
   ```bash
   # First-time setup
   ./scripts/init.sh
   ```

## Running the Simulator

The simulator runs in a Docker container with NVIDIA Isaac Lab:

```bash
# Start the simulator
./scripts/start.sh /isaaclab/CrazySim2Real/crazyflie_sim/run.py

# Alternative with custom parameters
./scripts/start.sh /isaaclab/CrazySim2Real/crazyflie_sim/run.py --port 8000
```

This will start a Docker container with the Isaac Lab environment and launch the Crazyflie simulator inside it. The simulator exposes an HTTP API that the benchmarking framework can connect to.

## Running Benchmarks

The benchmarking framework can run tests on either the simulator or real hardware:

```bash
# List available test plans
cd crazyflie_benchmark
python main.py --list-tests

# Run a benchmark in simulation
python main.py --config config/simulator_config.yaml --test-plan test_plans/step_tests.yaml

# Run a benchmark on real hardware
python main.py --config config/hardware_config.yaml --test-plan test_plans/step_tests.yaml
```

## Test Plans

Test plans are defined in YAML files in the `test_plans/` directory:

1. **Step Tests** (`step_tests.yaml`): Evaluates the system's response to step inputs in roll, pitch, and thrust
2. **Impulse Tests** (`impulse_tests.yaml`): Measures disturbance rejection and stability
3. **Sine Sweep Tests** (`sine_sweep_tests.yaml`): Evaluates frequency response characteristics

Example step test configuration:
```yaml
name: "Step Response Tests"
description: "Step response tests for roll, pitch, and thrust axes."
tests:
  - type: "step"
    channel: "roll"
    amplitude: 5.0  # degrees
    duration: 1.5   # seconds
  
  - type: "step"
    channel: "pitch"
    amplitude: 5.0  # degrees
    duration: 1.5   # seconds
```

## Analyzing Results

Test results are automatically logged in the `logs/` directory, with separate folders for simulation and real hardware tests. The framework provides tools for analyzing and visualizing test results:

```bash
# Analyze a specific test run
python main.py --analyze logs/20250513_001722-step-real/

# Generate plots and metrics
python main.py --analyze logs/20250513_001722-step-real/ --plots --metrics
```

## Sim-to-Real Comparison

The framework enables direct comparison between simulation and real hardware performance:

```bash
# Compare sim and real tests
python main.py --compare logs/sim/20250513_005534-step-sim/ logs/real/20250513_001722-step-real/
```

This will generate comparative metrics and plots showing the differences between simulation and reality.

## Configuration

Configuration files specify connection parameters and flight settings:

### Hardware Configuration (`hardware_config.yaml`)

```yaml
connection_type: "cflib"  # Use real Crazyflie hardware
uri: "radio://0/80/2M/E7E7E7E7E6"  # Radio URI for your Crazyflie
hover_thrust: 40000  # Approximate hover thrust (0-65535 scale)
max_roll_pitch: 15.0  # Safety limit for roll/pitch angle (degrees)
```

### Simulator Configuration (`simulator_config.yaml`)

```yaml
connection_type: "simulator"  # Use simulator
sim_host: "localhost"  # Simulator host address
sim_port: 8000  # Simulator port
hover_thrust: 0.6  # Approximate hover thrust (normalized 0-1 scale)
max_roll_pitch: 15.0  # Safety limit for roll/pitch angle (degrees)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Bitcraze](https://www.bitcraze.io/) for the Crazyflie platform
- [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac-sim) for the simulation environment
