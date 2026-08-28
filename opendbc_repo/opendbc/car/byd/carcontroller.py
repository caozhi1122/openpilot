import numpy as np

from opendbc.can.packer import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.byd.values import CarControllerParams, SongCarControllerParams, PLATFORM_SONG_PLUS_DMI
from opendbc.car.byd import bydcan
from opendbc.car.byd.carstate import STEER_TEMPLATE_DEFAULT
from opendbc.car.lateral import apply_driver_steer_torque_limits, apply_std_steer_angle_limits
from opendbc.car.interfaces import CarControllerBase

LongCtrlState = structs.CarControl.Actuators.LongControlState


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.CP = CP
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)
    # SONG PLUS DMI family uses the torque-based reference implementation
    self.is_song = CP.carFingerprint in PLATFORM_SONG_PLUS_DMI

    self.apply_angle_last = 0.0
    self.acc_idx = 0

    if self.is_song:
      # --- SONG PLUS DMI controller state (opendbc-master reference) ---
      self.last_steer_frame = 0
      self.last_acc_frame = 0
      self.apply_torque_last = 0
      self.mpc_lkas_counter = 0
      self.mpc_acc_counter = 0
      self.eps_fake318_counter = 0
      self.lkas_req_prepare = 0
      self.lkas_active = 0
      self.lat_safeoff = 0
      self.steer_softstart_limit = 0
      self.steerRateLimActive = False
      self.steerRateLim = 1.0
      self.first_start = True
      self.rfss = 0  # resume from standstill
      self.sss = 0   # standstill state
      self.apply_accel_last = 0

  def update(self, CC, CC_SP, CS, now_nanos):
    if self.is_song:
      return self._update_song(CC, CC_SP, CS, now_nanos)

    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # === STEERING ===
    # The EPS is a position servo: STEER_ANGLE is an absolute wheel angle target.
    # The command must therefore stay anchored to the measured angle at all times —
    # a limiter that only tracks its own previous output can ratchet away from the
    # wheel, saturate at the clamp and get every frame rejected by panda.
    if self.frame % self.params.STEER_STEP == 0:
      apply_angle = apply_std_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last,
                                                 CS.out.vEgoRaw, CS.out.steeringAngleDeg,
                                                 CC.latActive, self.params.ANGLE_LIMITS)

      # Hand control back to the driver rather than fighting them. DRIVER_EPS_TORQUE is
      # an unsigned magnitude from the column sensor, so this is a pure override test.
      if CS.out.steeringTorque > self.params.STEER_DRIVER_ALLOWANCE:
        apply_angle = CS.out.steeringAngleDeg

      # Windup guard: never let the command drift outside a fixed window around the
      # measured angle. Makes the saturation failure mode structurally impossible.
      apply_angle = float(np.clip(apply_angle,
                                  CS.out.steeringAngleDeg - self.params.MAX_ANGLE_ERROR,
                                  CS.out.steeringAngleDeg + self.params.MAX_ANGLE_ERROR))

      self.apply_angle_last = apply_angle

      # Transmit only while actually steering. Whenever we go quiet, panda stops
      # blocking the camera's command after ~150ms and the stock LKAS takes the wheel
      # back — so there is never a window with nobody driving the EPS. It also lets the
      # camera keep refreshing the frame template we copy.
      if CC.latActive:
        # Everything in the frame other than the angle is held at the constant the
        # camera itself sends while steering (bytes 0-2 = 2b 55 eb, at every speed).
        # It must not be varied: walking these fields across their range is what made
        # the car drop its whole ADAS mid-drive. Use the camera's own latched value
        # when we have seen it steer, and the captured constant otherwise — a blocked
        # lens means the camera never steers and never gives us one.
        template = CS.steer_template or STEER_TEMPLATE_DEFAULT

        can_sends.append(bydcan.create_steering_control(self.packer, apply_angle,
                                                        template,
                                                        self.frame // self.params.STEER_STEP))

        # Tell the cluster openpilot has the wheel. The camera drops its own LKAS
        # within seconds of us blocking its steering command, so without this the
        # LKAS indicator goes dark while openpilot is still steering. Sent on the
        # same cadence as the steering command, so panda hands 0x316 back to the
        # camera at the same moment it hands back 0x1E2.
        can_sends.append(bydcan.create_lkas_hud(self.packer, CS.lkas_hud,
                                                self.frame // self.params.STEER_STEP))

    # === LONGITUDINAL ===
    if self.CP.openpilotLongitudinalControl:
      if self.frame % self.params.STEER_STEP == 0:
        accel = float(np.clip(actuators.accel, self.params.ACCEL_MIN, self.params.ACCEL_MAX))
        if not CC.longActive or pcm_cancel_cmd:
          accel = 0.0

        can_sends.append(bydcan.create_acc_control(self.packer, accel,
                                                   CC.longActive and not pcm_cancel_cmd, self.acc_idx))

        set_speed = hud_control.setSpeed if hud_control.setSpeed > 0 else CS.out.cruiseState.speed
        can_sends.append(bydcan.create_acc_hud(self.packer, CC.enabled, set_speed * 3.6,
                                               hud_control.leadVisible, self.acc_idx))
        self.acc_idx += 1

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends

  def _update_song(self, CC, CC_SP, CS, now_nanos) -> tuple[structs.CarControl.Actuators, list]:
    """SONG PLUS DMI controller (ported from the opendbc-master reference).

    openpilot impersonates the camera: it replays ACC_MPC_STATE (0x316) on the ESC bus
    with LKAS_Output carrying the torque command, and sends a fake ACC_EPS_STATE (0x318)
    back to the camera bus so the stock MPC believes the EPS is alive (no DTC, AEB keeps
    working). ACC_CMD is replayed the same way when longitudinal is enabled.
    """
    P = SongCarControllerParams
    can_sends = []

    if (self.frame - self.last_steer_frame) >= P.STEER_STEP:
      # resolve counter mismatch problem
      if self.first_start:
        self.mpc_lkas_counter = int(CS.acc_mpc_state_counter + 1) & 0xF
        self.mpc_acc_counter = int(CS.acc_cmd_counter + 1) & 0xF
        self.eps_fake318_counter = int(CS.eps_state_counter + 1) & 0xF
        self.first_start = False

      apply_torque = 0

      if CC.latActive:
        if self.lkas_active:
          steer_desire = CC.actuators.torque

          if P.USE_STEERING_SPEED_LIMITER:  # use steering angular speed limiter
            rate_limit = np.interp(CS.out.aEgo, [8.3, 27.8], [132, 64])
            delta_rate = CS.steeringRateDegAbs - rate_limit

            if delta_rate < 0:
              self.steerRateLim -= 0.005 * delta_rate

              if delta_rate < -0.05:
                self.steerRateLimActive = False

              if self.steerRateLim > 1.0:
                self.steerRateLim = 1.0
                self.steerRateLimActive = False

            else:
              if self.steerRateLimActive:
                self.steerRateLim -= 0.005 * delta_rate
              else:
                self.steerRateLim = steer_desire
                self.steerRateLimActive = True

              if self.steerRateLim < 0:
                self.steerRateLim = 0

            new_steer_pu = np.clip(steer_desire, -self.steerRateLim, self.steerRateLim)
          else:
            new_steer_pu = steer_desire

          new_steer = int(round(new_steer_pu * P.STEER_MAX))

          if self.steer_softstart_limit < P.STEER_MAX:
            self.steer_softstart_limit = self.steer_softstart_limit + P.STEER_SOFTSTART_STEP
            new_steer = np.clip(new_steer, -self.steer_softstart_limit, self.steer_softstart_limit)

          apply_torque = apply_driver_steer_torque_limits(new_steer, self.apply_torque_last,
                                                          CS.out.steeringTorque, P)

        else:
          if CS.lkas_prepared:
            self.lkas_active = 1.0
            self.steerRateLimActive = False
            self.steerRateLim = 1.0
            self.lkas_req_prepare = 0
            self.steer_softstart_limit = 0
            self.lat_safeoff = 1
          else:
            self.lkas_req_prepare = 1

      elif self.lat_safeoff:
        if self.apply_torque_last == 0:
          self.lat_safeoff = 0
        apply_torque = apply_driver_steer_torque_limits(0, self.apply_torque_last,
                                                        CS.out.steeringTorque, P)

      else:
        self.lkas_req_prepare = 0
        self.steerRateLimActive = False
        self.steerRateLim = 1.0
        self.lkas_active = 0
        self.steer_softstart_limit = 0

      self.apply_torque_last = apply_torque

      self.mpc_lkas_counter = int(self.mpc_lkas_counter + 1) & 0xF
      self.eps_fake318_counter = int(self.eps_fake318_counter + 1) & 0xF
      self.last_steer_frame = self.frame

      # send steering command, op -> esc
      can_sends.append(bydcan.create_mpc_steering_control(self.packer, self.CP, CS.cam_lkas,
                                                          self.apply_torque_last, self.lkas_req_prepare, self.lkas_active,
                                                          CC.hudControl, self.mpc_lkas_counter))

      # send fake 0x318 from op -> mpc
      can_sends.append(bydcan.create_fake_318(self.packer, self.CP, CS.esc_eps,
                                              CS.mpc_laks_output, CS.mpc_laks_reqprepare, CS.mpc_laks_active,
                                              True, self.eps_fake318_counter))

    if (self.frame + 1 - self.last_acc_frame) >= P.ACC_STEP:
      accel = np.clip(CC.actuators.accel, P.ACCEL_MIN, P.ACCEL_MAX)

      if CC.longActive:
        stopping = CC.actuators.longControlState == LongCtrlState.stopping
        starting = CC.actuators.longControlState == LongCtrlState.starting
        running = CC.actuators.longControlState == LongCtrlState.pid

        # stopping and stopped
        if stopping and accel < -0.1:  # and CS.mrr_leading_dist < 4:
          self.rfss = 0
          self.sss = CS.out.standstill

        # re-starting
        elif starting and accel > 0.1 and CS.mrr_leading_dist > 3:
          self.rfss = CS.out.standstill
          self.sss = 0

        # started
        elif running:
          self.rfss = 0
          self.sss = 0

      else:
        accel = 0
        self.sss = 0
        self.rfss = 0

      can_sends.append(bydcan.acc_cmd(self.packer, self.CP, CS.cam_acc, CS.mrr_leading_dist,
                                      accel, self.rfss, self.sss, CC.longActive))

      self.apply_accel_last = accel
      self.last_acc_frame = self.frame + 1

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / P.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.accel = float(self.apply_accel_last)
    new_actuators.steeringAngleDeg = float(CS.out.steeringAngleDeg)

    self.frame += 1
    return new_actuators, can_sends
