# Re-export robot configs for backward compatibility.
# Import from gear_sonic.envs.manager_env.robots.g1 or .h2 directly for new code.
from gear_sonic.envs.manager_env.robots.g1 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.h2 import *  # noqa: F401,F403

# ADDED BY imprint (glvov): Unitree H1-2 (NOT the Unitree H2 above).
from gear_sonic.envs.manager_env.robots.h1_2 import *  # noqa: F401,F403
