"""Concrete backends (demo fakes live here; real ones plug in elsewhere)."""

from gaussian_robot.backends.demo import FakeRenderer, ScriptedDemoVLM

__all__ = ["FakeRenderer", "ScriptedDemoVLM"]
