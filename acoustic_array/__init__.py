"""A two-microphone acoustic direction sensor.

Microphones in, bearings out. It knows its own geometry and nothing else: no
rooms, no seats, no people. See api.py for what it deliberately cannot tell you.

    from acoustic_array import AcousticArray, AcousticEvent

    with AcousticArray.synthetic(angle_degrees=-20.0) as array:
        for event in array.stream():
            print(event.to_dict())
"""

from acoustic_array.api import (
    SOURCE_HARDWARE,
    SOURCE_SYNTHETIC,
    AcousticArray,
    AcousticEvent,
)

__all__ = [
    "AcousticArray",
    "AcousticEvent",
    "SOURCE_HARDWARE",
    "SOURCE_SYNTHETIC",
]
