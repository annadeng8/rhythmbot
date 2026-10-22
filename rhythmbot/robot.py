"""The Drake side: scene, hardware station, DiffIK controller, touch sensing.

Scene: the course's iiwa + WSG (`iiwa_and_wsg.dmd.yaml`), a floor, and a
touch screen slab in front of the arm (see `config.Layout` for the frame
conventions).  The screen carries *visual geometry only*: a position-
controlled arm pressing into a rigid welded box would just fight the
contact solver, and a touchscreen does not push back anyway.  Touch is
therefore sensed geometrically: a fingertip is "down" when it is within a
few millimetres of the screen plane, inside the screen.  The commanded
touch pose sits 2 mm below the plane so small tracking errors still count.

Controller: Drake's `DifferentialInverseKinematicsIntegrator` on a plant
holding only the arm (with the hand frozen onto it), fed the desired gripper
pose from the brain.  GuitarHeroBot used a bare pseudo-inverse and an
Integrator; this is the same idea with joint position and velocity limits
honored, which matters when a chart asks for fast lane changes.

Start pose: the arm is first put at a configuration that hovers over the
judgment line (solved with Drake's InverseKinematics against the course
default as the seed), and both the plant and the DiffIK integrator start
there.
"""

from __future__ import annotations

import os

import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    InverseKinematics,
    LeafSystem,
    Meshcat,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Solve,
    Sphere,
)
from pydrake.common import temp_directory

from .config import Config
from .planner import R_DOWN

IIWA_NAME = "iiwa"
WSG_NAME = "wsg"
GRIPPER_BODY = "body"


# ---------------------------------------------------------------------------
# Scene description
# ---------------------------------------------------------------------------
def make_screen_sdf(cfg: Config) -> str:
    """A visual-only slab with lane stripes and a judgment line."""
    L = cfg.layout
    size = L.screen_size()
    thickness = 0.02
    c = L.screen_center()
    visuals = [
        f"""
      <visual name="slab">
        <pose>0 0 {-thickness / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{size[0]:.4f} {size[1]:.4f} {thickness:.4f}</size></box></geometry>
        <material><diffuse>0.06 0.06 0.10 1</diffuse></material>
      </visual>"""
    ]
    for k in range(L.n_lanes + 1):
        x = -L.half_width + k * L.lane_width
        visuals.append(
            f"""
      <visual name="lane_line_{k}">
        <pose>{x:.4f} 0 0.0005 0 0 0</pose>
        <geometry><box><size>0.002 {L.highway_length + 0.04:.4f} 0.001</size></box></geometry>
        <material><diffuse>0.35 0.35 0.45 1</diffuse></material>
      </visual>"""
        )
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="touch_screen">
    <link name="screen">
      <pose>0 0 0 0 0 0</pose>
      {"".join(visuals)}
    </link>
    <frame name="screen_origin" attached_to="screen">
      <pose>0 0 0 0 0 0</pose>
    </frame>
  </model>
</sdf>
"""


def write_screen_sdf(cfg: Config) -> str:
    path = os.path.join(temp_directory(), "rhythmbot_touch_screen.sdf")
    with open(path, "w") as f:
        f.write(make_screen_sdf(cfg))
    return path


def make_scenario_yaml(cfg: Config, screen_sdf_path: str) -> str:
    c = cfg.layout.screen_center()
    return f"""
directives:
- add_directives:
    file: package://manipulation/iiwa_and_wsg.dmd.yaml
- add_model:
    name: floor
    file: package://manipulation/floor.sdf
- add_weld:
    parent: world
    child: floor::box
    X_PC:
        translation: [0, 0, -0.5]
- add_model:
    name: touch_screen
    file: file://{screen_sdf_path}
- add_weld:
    parent: world
    child: touch_screen::screen
    X_PC:
        translation: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]
plant_config:
    time_step: {cfg.plant_time_step}
model_drivers:
    {IIWA_NAME}: !IiwaDriver
        hand_model_name: {WSG_NAME}
    {WSG_NAME}: !SchunkWsgDriver {{}}
"""


# ---------------------------------------------------------------------------
# Station and controller
# ---------------------------------------------------------------------------
class Robot:
    """Everything built onto the DiagramBuilder for the arm."""

    def __init__(self, builder: DiagramBuilder, cfg: Config, meshcat: Meshcat | None):
        self.cfg = cfg
        screen_sdf = write_screen_sdf(cfg)
        self.scenario = LoadScenario(data=make_scenario_yaml(cfg, screen_sdf))
        self.station = builder.AddSystem(MakeHardwareStation(self.scenario, meshcat=meshcat))
        self.station.set_name("station")
        self.plant = self.station.GetSubsystemByName("plant")
        self.iiwa = self.plant.GetModelInstanceByName(IIWA_NAME)
        self.wsg = self.plant.GetModelInstanceByName(WSG_NAME)

        # Controller plant: the arm alone, with the hand welded on so its
        # `body` frame exists to be controlled.
        self.control_plant = MakeMultibodyPlant(
            self.scenario, model_instance_names=[IIWA_NAME], add_frozen_child_instances=True
        )
        assert self.control_plant.num_positions() == 7, self.control_plant.num_positions()
        self.G_control = self.control_plant.GetFrameByName(GRIPPER_BODY)

        params = DifferentialInverseKinematicsParameters(7, 7)
        params.set_joint_position_limits(
            (self.control_plant.GetPositionLowerLimits(), self.control_plant.GetPositionUpperLimits())
        )
        params.set_joint_velocity_limits(
            (self.control_plant.GetVelocityLowerLimits(), self.control_plant.GetVelocityUpperLimits())
        )
        self.diffik = builder.AddSystem(
            DifferentialInverseKinematicsIntegrator(
                self.control_plant, self.G_control, cfg.diffik_time_step, params
            )
        )
        self.diffik.set_name("diffik")
        self.params = self.diffik.get_mutable_parameters()
        builder.Connect(
            self.diffik.GetOutputPort("joint_positions"),
            self.station.GetInputPort(f"{IIWA_NAME}.position"),
        )
        self.q_start: np.ndarray | None = None

    # -- start configuration --------------------------------------------------
    def solve_start_configuration(self, X_WG: RigidTransform) -> np.ndarray:
        """Joint angles putting the gripper frame at X_WG, seeded from the
        scenario's default configuration."""
        ik = InverseKinematics(self.control_plant)
        ctx = ik.context()
        q_seed = self.control_plant.GetPositions(ctx)
        W = self.control_plant.world_frame()
        ik.AddPositionConstraint(
            self.G_control, [0, 0, 0], W, X_WG.translation() - 1e-3, X_WG.translation() + 1e-3
        )
        ik.AddOrientationConstraint(self.G_control, RotationMatrix(), W, X_WG.rotation(), 0.01)
        prog = ik.prog()
        prog.AddQuadraticErrorCost(np.eye(7), q_seed, ik.q())
        prog.SetInitialGuess(ik.q(), q_seed)
        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError(f"start pose IK failed for {X_WG.translation()}")
        q = result.GetSolution(ik.q())
        self.q_start = q
        self.params.set_nominal_joint_position(q)
        self.params.set_joint_centering_gain(0.5 * np.eye(7))
        return q

    def gripper_pose_at(self, q: np.ndarray) -> RigidTransform:
        ctx = self.control_plant.CreateDefaultContext()
        self.control_plant.SetPositions(ctx, q)
        return self.control_plant.CalcRelativeTransform(
            ctx, self.control_plant.world_frame(), self.G_control
        )

    # -- after Build() ---------------------------------------------------------
    def initialize(self, root_context) -> None:
        """Put the simulated arm and the DiffIK integrator at q_start, and fix
        the station inputs the brain does not drive."""
        assert self.q_start is not None, "call solve_start_configuration first"
        plant_ctx = self.plant.GetMyMutableContextFromRoot(root_context)
        self.plant.SetPositions(plant_ctx, self.iiwa, self.q_start)
        self.diffik.SetPositions(self.diffik.GetMyMutableContextFromRoot(root_context), self.q_start)
        station_ctx = self.station.GetMyMutableContextFromRoot(root_context)
        self.station.GetInputPort(f"{IIWA_NAME}.torque").FixValue(station_ctx, np.zeros(7))


# ---------------------------------------------------------------------------
# Touch sensing
# ---------------------------------------------------------------------------
# Output layout of TouchSensor: for each fingertip (left, right) a block of
# [down, x, y, z].
TOUCH_WIDTH = 8


class TouchSensor(LeafSystem):
    """Fingertip positions from the station's body poses, and whether each
    tip is on the screen.  Draws the tips in Meshcat (red when down)."""

    def __init__(self, plant, cfg: Config, meshcat: Meshcat | None):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.layout
        self.meshcat = meshcat
        self.finger_indices = [
            plant.GetBodyByName("left_finger").index(),
            plant.GetBodyByName("right_finger").index(),
        ]
        self.p_Ftip = np.array([0.0, cfg.gripper.finger_tip_offset, 0.0])
        self.poses_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareVectorOutputPort("touch", TOUCH_WIDTH, self._calc)
        if meshcat is not None:
            self.DeclarePeriodicPublishEvent(1.0 / cfg.visualizer_hz, 0.0, self._draw)
            for i in range(2):
                meshcat.SetObject(f"/fingertips/{i}", Sphere(0.006), Rgba(0.2, 0.9, 0.2, 0.9))
        half = 0.5 * self.L.screen_size()
        c = self.L.screen_center()
        self.x_range = (c[0] - half[0], c[0] + half[0])
        self.y_range = (c[1] - half[1], c[1] + half[1])

    def tips(self, context) -> np.ndarray:
        poses = self.poses_port.Eval(context)
        return np.array([poses[int(i)] @ self.p_Ftip for i in self.finger_indices])

    def _calc(self, context, output):
        out = np.zeros(TOUCH_WIDTH)
        cfg_tol = self.cfg.motion.contact_tolerance
        for k, p in enumerate(self.tips(context)):
            down = (
                p[2] <= self.L.screen_z + cfg_tol
                and self.x_range[0] <= p[0] <= self.x_range[1]
                and self.y_range[0] <= p[1] <= self.y_range[1]
            )
            out[4 * k : 4 * k + 4] = [float(down), p[0], p[1], p[2]]
        output.SetFromVector(out)

    def _draw(self, context):
        vals = self._calc_values(context)
        t = context.get_time()
        for k in range(2):
            down, x, y, z = vals[4 * k : 4 * k + 4]
            self.meshcat.SetTransform(f"/fingertips/{k}", RigidTransform([x, y, z]), t)
            self.meshcat.SetProperty(
                f"/fingertips/{k}", "color", [0.9, 0.15, 0.15, 0.9] if down else [0.2, 0.9, 0.2, 0.9]
            )

    def _calc_values(self, context) -> np.ndarray:
        return self.get_output_port(0).Eval(context)


def unpack_touch(vals: np.ndarray) -> list[tuple[bool, np.ndarray]]:
    """[(down, p_Wtip), (down, p_Wtip)] from a TouchSensor output vector."""
    return [(bool(vals[4 * k] > 0.5), vals[4 * k + 1 : 4 * k + 4]) for k in range(2)]
