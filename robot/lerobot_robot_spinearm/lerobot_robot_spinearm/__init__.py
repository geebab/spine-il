"""lerobot_robot_spinearm — SO-101 arm server + LeRobot proxy robots `spine_arm` / `spine_bi_arm`.

Importing this package registers the robot types with LeRobot (the config module runs the
`register_subclass` decorators) and exposes the classes so LeRobot can resolve
`SpineArmConfig` → `SpineArm` in this package.
"""

from .client import ArmError, ArmRpc, ArmTimeout, SpineArm, SpineBiArm, watch
from .config_spinearm import SpineArmConfig, SpineBiArmConfig

__all__ = ["ArmError", "ArmRpc", "ArmTimeout", "SpineArm", "SpineArmConfig", "SpineBiArm", "SpineBiArmConfig", "watch"]
__version__ = "0.1.0"
