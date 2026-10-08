"""The flame, sprays, laser and thrown things start at the nozzle of the tool you're holding, on screen and in front
of you. Before 2.7.2 they were placed 0.7 m behind your head and flew through the camera."""
import math

import numpy as np
import pytest


@pytest.mark.parametrize("fov", [50.0, 70.0, 110.0])
@pytest.mark.parametrize("tool", ["torch", "cleaner", "freeze", "laser", "zapper"])
def test_tool_tip_matches_the_drawn_nozzle(fov, tool):
    from types import SimpleNamespace

    from kickthefly.game import kick3d
    from kickthefly.game.kick_the_fly import TOOLS

    player = kick3d.Player()
    host = SimpleNamespace(player=player, tool=[t[0] for t in TOOLS].index(tool), cfg={"controls.fov": fov})
    tip = kick3d.Game3D.tool_tip(host)

    rel = tip - player.eye
    fwd = player.forward()
    assert rel @ fwd > 0.3, "the tip must be in front of you"

    # screen position of the world point, through your field of view
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = math.tan(math.radians(fov) / 2)
    world_xy = np.array([rel @ right, rel @ up]) / (rel @ fwd) / t
    # screen position of the drawn nozzle, through the viewmodel's own 60 degree lens
    bob = 0.012 * math.sin(player.walk_phase * 2)
    nozzle = kick3d.VIEW_BASE + (0.0, bob, 0.0) + kick3d.NOZZLE[tool]
    view_xy = np.array([nozzle[0], nozzle[1]]) / (-nozzle[2]) / math.tan(math.radians(kick3d.VIEWMODEL_FOV) / 2)
    assert np.allclose(world_xy, view_xy, atol=1e-6)
