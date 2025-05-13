#!/usr/bin/python3

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
import re

def setup_environment():
    """Set up the environment variables and paths"""
    script_dir = Path(__file__).parent.absolute()
    # Add the script directory to Python path
    python_path = os.environ.get('PYTHONPATH', '')
    if str(script_dir) not in python_path:
        os.environ['PYTHONPATH'] = f"{python_path}:{script_dir}"
    return script_dir

def check_target_script(script_path):
    """Check if the target script exists"""
    if not os.path.isfile(script_path):
        print(f"Error: {script_path} not found")
        return False
    return True

def get_available_gpus():
    """Get the number of available GPUs and their IDs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        return gpu_ids
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Unable to query GPUs with nvidia-smi. Assuming no GPUs available.")
        return []

def get_idle_gpus():
    """Get the IDs of idle GPUs (GPUs without running processes)"""
    try:
        # Get GPU usage information
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        busy_uuids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        # Get all GPU UUIDs and their indices
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True)
        
        # Parse GPU index and UUID information
        all_gpus = {}
        for line in gpu_info.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    idx, uuid = parts
                    all_gpus[uuid] = int(idx)
        
        # Identify idle GPUs
        idle_gpu_ids = [idx for uuid, idx in all_gpus.items() if uuid not in busy_uuids]
        return idle_gpu_ids
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Unable to determine idle GPUs. Assuming all GPUs are busy.")
        return []

def wait_for_gpu(max_wait_minutes=30):
    """Wait for an available GPU, checking every 30 seconds for up to max_wait_minutes"""
    print(f"No idle GPUs found. Waiting for an available GPU (max {max_wait_minutes} minutes)...")
    
    start_time = time.time()
    end_time = start_time + (max_wait_minutes * 60)
    
    while time.time() < end_time:
        idle_gpus = get_idle_gpus()
        if idle_gpus:
            print(f"Found idle GPU(s): {idle_gpus}")
            return idle_gpus
        
        elapsed_minutes = (time.time() - start_time) / 60
        remaining_minutes = max_wait_minutes - elapsed_minutes
        print(f"Still waiting... {elapsed_minutes:.1f} minutes elapsed, {remaining_minutes:.1f} minutes remaining")
        time.sleep(30)  # Check every 30 seconds
    
    print(f"Waited {max_wait_minutes} minutes but no GPU became available.")
    return []

def select_gpu(requested_gpus):
    """Select GPU(s) based on availability and request"""
    if requested_gpus is not None:
        print(f"Using user-specified GPU(s): {requested_gpus}")
        return requested_gpus
    
    # Auto GPU scheduling
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No GPUs detected in the system")
        return None
    
    if len(available_gpus) == 1:
        print(f"System has only one GPU (ID: {available_gpus[0]}), using it directly")
        return str(available_gpus[0])
    
    # Multiple GPUs available, check which ones are idle
    idle_gpus = get_idle_gpus()
    
    if idle_gpus:
        selected_gpu = str(idle_gpus[0])
        print(f"Selected idle GPU: {selected_gpu}")
        return selected_gpu
    else:
        # No idle GPUs, wait for one to become available
        idle_gpus = wait_for_gpu(30)  # Wait up to 30 minutes
        if idle_gpus:
            selected_gpu = str(idle_gpus[0])
            print(f"Selected GPU after waiting: {selected_gpu}")
            return selected_gpu
        else:
            print("No GPU became available after waiting. Proceeding without specific GPU selection.")
            return None

def run_isaac_lab(script_dir, target_script, script_args, gui, gpus):
    """Run the target script with Isaac Lab"""
    isaaclab_sh = script_dir.parent.parent / "isaaclab.sh"
    headless_args = ["--headless", "--livestream", "2"]
    gui = True  # Force GUI mode for now
    # headless_args = ["--headless"]
    if gui:
        headless_args = []
    else:
        os.environ["DISPLAY"] = ":0"

    # Apply GPU selection logic
    selected_gpus = select_gpu(gpus)
    if selected_gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpus
        print(f"Setting CUDA_VISIBLE_DEVICES={selected_gpus}")

    # Build command
    cmd = [
        str(isaaclab_sh),
        "-p",
        target_script,
        *script_args,
        *headless_args,
        "--enable_camera",
        "--kit_args",
        # "--enable isaacsim.asset.gen.omap", # Do not split this line
        "--enable omni.kit.livestream.webrtc --enable isaacsim.asset.gen.omap", # Do not split this line
    ]

    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Process interrupted by user")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Run script with Isaac Lab")
    parser.add_argument("--gui", action="store_true", help="Run with GUI")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU indices to use")
    parser.add_argument("target_script", help="Path to the target script to run")
    parser.add_argument("script_args", nargs="*",
                        help="Arguments to pass to the target script")

    # Handle script arguments that might be confused with this script's arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        parser.print_help()
        sys.exit(0)

    # If no arguments provided, show usage
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args, unknown = parser.parse_known_args()

    # Combine explicitly parsed arguments with any unknown arguments
    all_script_args = args.script_args + unknown

    script_dir = setup_environment()

    target_script = args.target_script
    if not check_target_script(target_script):
        sys.exit(1)

    run_isaac_lab(script_dir, target_script, all_script_args, args.gui, args.gpus)

if __name__ == "__main__":
    main()