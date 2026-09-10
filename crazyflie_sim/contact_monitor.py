"""Latched contact telemetry; no simulator imports or flight commands."""

from copy import deepcopy


class ContactMonitor:
    def __init__(self):
        self.active = False
        self.sequence = 0
        self.last_event = None

    def update(self, active, timestamp, previous_state, current_state):
        if active and not self.active:
            self.sequence += 1
            self.last_event = {
                "sequence": self.sequence, "timestamp": float(timestamp),
                "before": deepcopy(previous_state), "position": deepcopy(current_state["position"]),
            }
        self.active = bool(active)

    def reset(self):
        # Keep the event visible across a reset until a client has observed it.
        self.active = False

    def status(self):
        return {"supported": True, "active": self.active, "sequence": self.sequence,
                "last_event": deepcopy(self.last_event)}
