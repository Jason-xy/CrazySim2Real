#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if running inside Docker (use HOST_BASE_DIR if available)
if [ -n "${HOST_BASE_DIR:-}" ]; then
    # Remove the first-level directory from SCRIPT_DIR
    echo "Running inside Docker, using HOST_BASE_DIR: $HOST_BASE_DIR"
    SCRIPT_DIR_NO_FIRST=$(echo "$(dirname "$SCRIPT_DIR")" | cut -d'/' -f3-)
    export BASE_DIR="$HOST_BASE_DIR/$SCRIPT_DIR_NO_FIRST"
else
    # Regular path resolution when not in Docker
    export BASE_DIR="$(dirname "$SCRIPT_DIR")"
    echo "Using local BASE_DIR: $BASE_DIR"
fi

# Generate a unique project name based on the current timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)
PROJECT_NAME="isaaclab_${TIMESTAMP}_$$"
DOCKER_COMPOSE_FILE="$(dirname "$SCRIPT_DIR")/docker/isaaclab/docker-compose.yml"
COMPOSE=(docker compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME")
XHOST_ADDED=false
CONTAINER_STARTED=false

# Create shared volumes if they don't exist yet
create_shared_volumes() {
    # Check for shared volumes and create them if they don't exist
    echo "Setting up shared volumes..."
    
    # List of volumes to check/create
    SHARED_VOLUMES=(
        "isaac_shared_cache_kit"
        "isaac_shared_cache_ov"
        "isaac_shared_cache_pip"
        "isaac_shared_cache_gl"
        "isaac_shared_cache_compute"
        "isaac_shared_logs"
        "isaac_shared_carb_logs"
        "isaac_shared_data"
        "isaac_shared_docs"
        "isaac_shared_lab_docs"
        "isaac_shared_lab_logs"
        "isaac_shared_lab_data"
    )
    
    for VOLUME in "${SHARED_VOLUMES[@]}"; do
        if ! docker volume inspect "$VOLUME" &>/dev/null; then
            echo "Creating volume: $VOLUME"
            docker volume create "$VOLUME"
        fi
    done
    
    # Set environment variables to map volumes
    export ISAAC_CACHE_KIT=isaac_shared_cache_kit
    export ISAAC_CACHE_OV=isaac_shared_cache_ov
    export ISAAC_CACHE_PIP=isaac_shared_cache_pip
    export ISAAC_CACHE_GL=isaac_shared_cache_gl
    export ISAAC_CACHE_COMPUTE=isaac_shared_cache_compute
    export ISAAC_LOGS=isaac_shared_logs
    export ISAAC_CARB_LOGS=isaac_shared_carb_logs
    export ISAAC_DATA=isaac_shared_data
    export ISAAC_DOCS=isaac_shared_docs
    export ISAAC_LAB_DOCS=isaac_shared_lab_docs
    export ISAAC_LAB_LOGS=isaac_shared_lab_logs
    export ISAAC_LAB_DATA=isaac_shared_lab_data
}

# Function to clean up containers when script exits
cleanup() {
    if $CONTAINER_STARTED; then
        "${COMPOSE[@]}" down --remove-orphans || true
    fi
    if $XHOST_ADDED; then
        xhost -si:localuser:root >/dev/null || true
    fi
}

# Register the cleanup function to run on script exit
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Display help message
show_help() {
    echo "Usage: ./start.sh [options] [command]"
    echo ""
    echo "This script starts a Docker container with Isaac Sim environment."
    echo ""
    echo "Options:"
    echo "  -h, --help     Display this help message and exit"
    echo "  --stop-all     Stop all running containers created by this script"
    echo ""
    echo "If command arguments are provided, they will be passed to the container's entrypoint."
    echo "Example: ./start.sh <custom_script.py> <arg1> <arg2> ..."
    echo ""
    echo "If no arguments are provided, the container will use its default entrypoint."
    echo ""
    echo "Each run creates a new container instance with a unique name."
    # Don't call exit directly as it would trigger cleanup
    trap - EXIT INT TERM
    exit 0
}

# Function to stop all containers
stop_all_containers() {
    echo "Stopping all isaaclab containers..."
    # Find all project names starting with isaaclab_
    PROJECTS=$(docker compose ls --format "{{.Project}}" | grep "^isaaclab_" || true)
    
    if [ -z "$PROJECTS" ]; then
        echo "No running isaaclab containers found."
    else
        for PROJECT in $PROJECTS; do
            echo "Stopping project: $PROJECT"
            docker compose -p $PROJECT down
        done
        echo "All isaaclab containers stopped."
    fi
    # Don't call exit directly as it would trigger cleanup for the current container too
    trap - EXIT INT TERM
    exit 0
}

# Check for help option or stop-all command
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
elif [ "${1:-}" = "--stop-all" ]; then
    stop_all_containers
fi

# Keep mounted source aligned with the shared runtime release.
EXPECTED=ffff603eafc6b74264a5261cc0183d6a65390d78
export ISAACLAB_IMAGE=crazyrl-isaaclab:3.0.0-beta2.patch1
export ISAACSIM_VERSION=6.0.1
LAB_DIR="$(dirname "$SCRIPT_DIR")/docker/isaaclab/IsaacLab"
if [ "$(git -C "$LAB_DIR" rev-parse HEAD)" != "$EXPECTED" ]; then
    echo "Isaac Lab must be checked out at v3.0.0-beta2.patch1 ($EXPECTED)." >&2
    exit 1
fi
if [ -n "$(git -C "$LAB_DIR" status --porcelain --untracked-files=no)" ]; then
    echo "Isaac Lab has local source changes; refusing to label them as the pinned release." >&2
    exit 1
fi
VIZ=$(python3 "$SCRIPT_DIR/run.py" --print-viz "$@")
if [[ ",$VIZ," == *",kit,"* ]]; then
    if [ -z "${DISPLAY:-}" ]; then
        echo "GUI needs DISPLAY. For automatic recording pass --viz none." >&2
        exit 1
    fi
    XHOST_STATE=$(xhost)
    if ! grep -qi 'access control disabled' <<< "$XHOST_STATE" &&
       ! grep -qi 'SI:localuser:root' <<< "$XHOST_STATE"; then
        xhost +si:localuser:root
        XHOST_ADDED=true
    fi
fi

create_shared_volumes
mkdir -p "$BASE_DIR/logs"
# Reuse the CrazyE2E image without rebuilding its shared tag on every launch.
if ! docker image inspect "$ISAACLAB_IMAGE" >/dev/null 2>&1; then
    "${COMPOSE[@]}" build crazyesim2real
fi
echo "Starting new container with project name: $PROJECT_NAME"
CONTAINER_STARTED=true
if [ $# -gt 0 ]; then
    "${COMPOSE[@]}" run --rm -T --name "${PROJECT_NAME}-sim" \
        --entrypoint /bin/bash crazyesim2real \
        /workspace/isaaclab/CrazySim2Real/scripts/isaac_python.sh \
        /workspace/isaaclab/CrazySim2Real/scripts/run.py "$@"
else
    "${COMPOSE[@]}" run --rm --name "${PROJECT_NAME}-sim" --entrypoint bash crazyesim2real
fi
