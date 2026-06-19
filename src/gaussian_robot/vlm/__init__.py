"""Vision-Language Model client interface."""

from gaussian_robot.vlm.client import Decision, VLMClient
from gaussian_robot.vlm.observation import Observation

__all__ = ["Decision", "Observation", "VLMClient"]
