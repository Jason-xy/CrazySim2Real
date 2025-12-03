# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrazySim2Real is a comprehensive sim2real benchmarking framework for Crazyflie drones that enables evaluation of control algorithms in both simulation and real hardware environments. The project consists of two main components:

1. **Crazyflie Simulator** (`crazyflie_sim/`) - Physics-based drone simulator built on NVIDIA Isaac Lab
2. **Benchmarking Framework** (`crazyflie_benchmark/`) - Unified testing interface for sim and real hardware

## Development Commands

### Initial Setup
```bash
# Initialize git submodules (required for Isaac Lab)
./scripts/init.sh

# Install Python dependencies for benchmarking
pip install -r crazyflie_benchmark/requirements.txt
```

### Running the Simulator
```bash
# Start simulator with Docker (creates unique container instance)
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/run.py

# Start simulator with custom parameters
./scripts/start.sh /workspace/isaaclab/CrazySim2Real/crazyflie_sim/run.py --port 8000

# Stop all running simulator containers
./scripts/start.sh --stop-all
```

### Running Benchmarks
```bash
cd crazyflie_benchmark

# List available test plans
python main.py --list-tests

# Run tests in simulation
python main.py --config config/simulator_config.yaml --test-plan step_tests.yaml

# Run tests on real hardware
python main.py --config config/hardware_config.yaml --test-plan step_tests.yaml

# Analyze test results
python main.py --analyze logs/20250513_001722-step-real/

# Compare sim vs real results
python main.py --compare logs/sim/20250513_005534-step-sim/ logs/real/20250513_001722-step-real/
```

## Architecture Overview

### Core Components

**crazyflie_benchmark/** - Main testing framework
- `main.py` - CLI entry point with comprehensive argument parsing
- `controller.py` - Flight controller interface for both sim and hardware
- `connection.py` - Connection management for hardware/simulator
- `safety.py` - Safety monitoring system  
- `logging_manager.py` - Data logging and persistence
- `data_processor.py` - Post-processing and metrics calculation
- `plotter.py` - Visualization and plotting utilities
- `tests/` - Test implementations (StepTest, ImpulseTest, SineSweepTest)
- `config/` - Configuration files for hardware and simulator

**crazyflie_sim/** - Isaac Lab-based simulator
- `sim/simulation_manager.py` - Core simulation orchestration with DroneState dataclass
- `controllers/` - Controller implementations (PID, attitude control)
- `api/` - HTTP API server for external communication
- `run.py` - Simulator entry point script

### Key Design Patterns

**Unified Interface**: Both simulator and hardware use identical test protocols through the same controller interface, enabling direct sim2real comparison.

**Safety-First Design**: Built-in safety monitoring prevents dangerous commands on real hardware with configurable limits.

**Modular Controllers**: Controller architecture supports easy swapping between different control strategies (PID, attitude control, custom implementations).

**Docker Integration**: Simulator runs in Isaac Lab Docker container with volume mounting and automatic cleanup.

### Data Flow

1. Test plans (YAML) define test sequences (step, impulse, sine sweep)
2. Main CLI loads config and instantiates appropriate connection (hardware/sim)
3. FlightController executes test commands through unified interface
4. SafetyMonitor validates all commands before execution
5. LoggingManager captures telemetry data during tests
6. DataProcessor analyzes results and generates metrics
7. Plotter creates visualization outputs

### Configuration System

- Hardware config: Crazyflie radio URI, hover thrust, safety limits
- Simulator config: Host/port, normalized thrust values
- Test plans: YAML files defining test parameters and sequences
- Safety limits: Configurable bounds for roll/pitch angles and thrust

### Thread Safety

The simulator uses thread-safe DroneState dataclass for state management between simulation thread and API server, ensuring consistent data access.