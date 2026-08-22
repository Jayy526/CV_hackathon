"""Legacy import path. The implementation moved to acoustic_array.packets.

Kept so heimdall.audio.* keeps working unchanged. Do not add anything here; new
work belongs in the acoustic_array package.
"""

from acoustic_array.packets import *  # noqa: F401,F403
