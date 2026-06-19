"""Robot navigation: actions, state, the exploration loop, and termination.

Note: :class:`Explorer` and :class:`ObservationBuilder` are imported via their
modules (``gaussian_robot.nav.explorer`` / ``gaussian_robot.nav.observation``)
to keep this package init free of heavy import cycles.
"""

from gaussian_robot.nav.action import Action, ActionSpace, apply_action
from gaussian_robot.nav.robot import Robot

__all__ = ["Action", "ActionSpace", "Robot", "apply_action"]
