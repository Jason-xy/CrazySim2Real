"""
Hover thrust measurement helper.

Collects thrust telemetry during a short hover hold and returns the average
hover thrust. Works for simulator and real runs as long as telemetry exposes
`stabilizer.thrust` (or `thrust`).
"""
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def measure_hover_thrust(controller, logger_obj, sample_time: float = 3) -> Optional[int]:
    """
    Measure hover thrust by holding hover and averaging logged thrust.

    Args:
        controller: FlightController instance (provides send_hover_setpoint, CONTROL_PERIOD)
        logger_obj: FlightLogger instance (provides get_current_position, get_latest_telemetry)
        sample_time: Duration to collect samples (seconds)

    Returns:
        int hover thrust on success, or None on failure.
    """
    pos = logger_obj.get_current_position()
    target_height = pos.get("z", 0.5)

    samples = []
    dt = controller.CONTROL_PERIOD
    end_time = time.monotonic() + sample_time
    next_tick = time.monotonic()

    while time.monotonic() < end_time:
        controller.send_hover_setpoint(0, 0, 0, target_height)
        telem = logger_obj.get_latest_telemetry()
        thrust_val = None
        for key in ("stabilizer.thrust", "thrust"):
            if key in telem:
                try:
                    thrust_val = float(telem[key])
                    break
                except (ValueError, TypeError):
                    continue
        if thrust_val is not None:
            samples.append(thrust_val)
        next_tick += dt
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.monotonic()

    if not samples:
        logger.error("Hover thrust measurement failed: no telemetry samples.")
        return None

    measured = int(sum(samples) / len(samples))
    logger.info(f"Measured hover thrust from logs: {measured}")
    return measured
