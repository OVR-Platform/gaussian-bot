"""Serialize live :class:`SessionEvent`s to JSON for the WebSocket client.

Image panels are JPEG-encoded as ``data:`` URLs (reusing the Qwen client's
encoder) so the browser can drop them straight into ``<img src>``.
"""

from __future__ import annotations

from typing import Any

from gaussian_robot.events import (
    MarkEvent,
    SceneDescribeEvent,
    SessionEndEvent,
    SessionStartEvent,
    StepEvent,
    WalkEndEvent,
)
from gaussian_robot.vlm.qwen import jpeg_data_url


def event_to_message(event: Any) -> dict[str, Any]:  # noqa: PLR0911 (one return per event type)
    if isinstance(event, SessionStartEvent):
        return {
            "type": "session_start",
            "bounds_min": event.bounds_min.tolist(),
            "bounds_max": event.bounds_max.tolist(),
            "up_axis": event.up_axis,
            "total_seeds": event.total_seeds,
            "requested_seeds": event.requested_seeds,
            "seeds": event.seed_floor.tolist(),
            "seed_kinds": event.seed_kinds,
            "frontiers": event.frontier_floor.tolist(),
        }
    if isinstance(event, StepEvent):
        panels = {label: jpeg_data_url(img) for label, img in event.observation.panels}
        return {
            "type": "step",
            "walk_id": event.walk_id,
            "step": event.step,
            "budget": event.budget,
            "action": event.action.value,
            "raw_text": event.decision.raw_text,
            "novelty": event.novelty,
            "degenerate": event.degenerate,
            "blocked": event.blocked,
            "pose": event.pose.position.tolist(),
            "coverage_floor": event.coverage_floor,
            "coverage_pose_space": event.coverage_pose_space,
            "panels": panels,
            "sampled": event.sampled_floor.tolist(),
            "trail": event.trail_floor.tolist(),
        }
    if isinstance(event, MarkEvent):
        return {
            "type": "mark",
            "walk_id": event.walk_id,
            "step": event.step,
            "floor": event.floor.tolist(),
            "count": event.count,
            "auto": event.auto,
        }
    if isinstance(event, WalkEndEvent):
        return {
            "type": "walk_end",
            "walk_id": event.walk_id,
            "reason": event.reason,
            "steps": event.steps,
        }
    if isinstance(event, SceneDescribeEvent):
        return {
            "type": "scene_describe",
            "walk_id": event.walk_id,
            "step": event.step,
            "description": event.description,
        }
    if isinstance(event, SessionEndEvent):
        return {
            "type": "session_end",
            "reason": event.reason,
            "total_steps": event.total_steps,
            "total_poses": event.total_poses,
        }
    return {"type": "unknown"}
