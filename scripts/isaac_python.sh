#!/bin/bash
# Set the SDK environment, then keep Python as PID 1 for graceful SIGTERM.
set -eo pipefail
ISAAC_ROOT="${ISAACSIM_PATH:-/workspace/isaaclab/_isaac_sim}"
export CARB_APP_PATH="$ISAAC_ROOT/kit"
export ISAAC_PATH="$ISAAC_ROOT"
export EXP_PATH="$ISAAC_ROOT/apps"
export RESOURCE_NAME=IsaacSim
source "$ISAAC_ROOT/setup_python_env.sh"
export LD_PRELOAD="$ISAAC_ROOT/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$ISAAC_ROOT/kit/python/bin/python3" "$@"
