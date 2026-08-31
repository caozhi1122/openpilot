import copy

import numpy as np

from opendbc.can import CANDefine
from opendbc.can.parser import CANParser
from opendbc.car import Bus, DT_CTRL, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.byd.values import DBC, BUTTONS, LKAS_HUD_PASSTHROUGH, LKASConfig, CanBus, PLATFORM_SONG_PLUS_DMI, SongCarControllerParams

ButtonType = structs.CarState.ButtonEvent.Type

# UNKNOWN (bytes 0-1), SET_ME_X01 and SET_ME_XE of STEERING_MODULE_ADAS are *constant* for
# the whole of a camera steering episode — the camera holds bytes 0-2 at 2b 55 eb from the
# first STEER_REQ frame to the last, and 00 00 e0 while idle. Measured over the 2026-08-09
# drive: 1111 of the 1121 frames the camera sent with STEER_REQ set carried exactly this
# triple, the remaining ten being the one- or two-frame ramp in and out of an episode.
#
# They are NOT a 20-bit sequence to be continued, and treating them as one is what broke
# lateral control: openpilot latched the camera's value and added a fixed step every frame,
# which walks SET_ME_XE through all sixteen values and SET_ME_X01 through all four. The car
# accepts a few tens of seconds of that and then drops the entire stock ADAS — ACC_HUD_ADAS
# ACC_ON1/ACC_ON2 and every ACC_CMD engagement bit go to zero together, taking openpilot's
# cruiseState with them. Only ever send a triple the camera itself has been seen to send.
STEER_TEMPLATE_FIELDS = ("UNKNOWN", "SET_ME_X01", "SET_ME_XE")

# The camera's steering frame, captured on-car: bytes 0-2 = 2b 55 eb.
STEER_TEMPLATE_DEFAULT = {"UNKNOWN": 2773, "SET_ME_X01": 1, "SET_ME_XE": 0xB}

# The camera ramps in over a frame or two at the start of an episode, so only adopt a
# template once it has held still — otherwise we can latch a transitional value.
STEER_TEMPLATE_STABLE_FRAMES = 5

# ACC_CMD arrives at 50Hz with no counter/checksum validation in the parser, and a
# single corrupted frame must not drop cruiseState.enabled for one frame — enough to fire
# pcmDisable and drop panda's controls_allowed. Require a few consecutive frames to drop.
CRUISE_DROP_FRAMES = 5


class CarState(CarStateBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)
    # SONG PLUS DMI family runs a completely different protocol (byd_song.dbc) than the
    # ATTO3: torque-based LKAS over ACC_MPC_STATE/ACC_EPS_STATE instead of the angle
    # STEERING_MODULE_ADAS. Both implementations share this class, dispatched per car.
    self.is_song = CP.carFingerprint in PLATFORM_SONG_PLUS_DMI

    self.button_states = {button.event_type: False for button in BUTTONS}
    self.lkas_hud = dict.fromkeys(LKAS_HUD_PASSTHROUGH, 0)
    # the camera's own steering frame, held constant; None until it has steered once
    self.steer_template = None
    self._template_candidate = None
    self._template_frames = 0
    self.acc_off_frames = 0
    self.cruise_enabled_last = False
    self.prev_angle = 0.0

    if self.is_song:
      # --- SONG PLUS DMI state (ported from the opendbc-master reference) ---
      can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])
      self.shifter_values = can_define.dv["DRIVE_STATE"]["Gear"]

      self.speed_kph = 0
      self.mpc_lkas_config = 0
      self.acc_hud_adas_counter = 0
      self.acc_mpc_state_counter = 0
      self.acc_cmd_counter = 0
      self.eps_warning = False
      self.acc_active_last = False
      self.low_speed_alert = False
      self.lkas_allowed_speed = False
      self.lkas_prepared = False  # 0x318 ACC_EPS_STATE, EPS -> OP
      self.acc_state = 0
      self.adas_set_dist = 0
      self.mpc_laks_output = 0
      self.mpc_laks_active = False
      self.mpc_laks_reqprepare = False
      self.cam_lkas = 0
      self.cam_acc = 0
      self.cam_adas = 0
      self.esc_eps = 0
      self.mrr_leading_dist = 0
      self.btn_acc_cancel = 0
      self.btn_acc_set_reset = 0
      self.btn_acc_dist_inc = 0
      self.btn_acc_dist_dec = 0
      self.prev_steeringAngleDeg = 0
      self.steeringRateDegAbs = 0

  def update_steer_template(self, cam_steer) -> None:
    """
        Latch the camera's steering frame while the camera is the one driving the EPS, so
        that what we transmit is byte-identical to what this car's own ADAS transmits.

        The fields are constant for the whole of a steering episode, so only adopt a value
        once it has held still — the camera spends a frame or two ramping in and out, and
        latching one of those transitional frames would have us send, forever, a triple the
        car never sends.
        """
    if not cam_steer["STEER_REQ"]:
      self._template_candidate = None
      self._template_frames = 0
      return

    template = {k: int(cam_steer[k]) for k in STEER_TEMPLATE_FIELDS}
    if not any(template.values()):  # the idle frame, sent during the ramp in
      return

    if template == self._template_candidate:
      self._template_frames += 1
    else:
      self._template_candidate = template
      self._template_frames = 1

    if self._template_frames >= STEER_TEMPLATE_STABLE_FRAMES:
      self.steer_template = template

  def update(self, can_parsers) -> tuple[structs.CarState, structs.CarStateSP]:
    if self.is_song:
      return self._update_song(can_parsers)

    cp = can_parsers[Bus.pt]       # bus 0: car-side chassis CAN
    cp_cam = can_parsers[Bus.cam]  # bus 2: camera-side chassis CAN
    ret = structs.CarState.new_message()

    # --- Steering ---
    ret.steeringAngleDeg = cp.vl["STEER_MODULE_2"]["STEER_ANGLE_2"]
    ret.steeringRateDeg = (ret.steeringAngleDeg - self.prev_angle) / DT_CTRL
    self.prev_angle = ret.steeringAngleDeg
    # DRIVER_EPS_TORQUE (byte 2 of STEER_MODULE_2): actual column torque sensor, raw 0–255 unsigned.
    # MAIN_TORQUE (STEERING_TORQUE 0x1FC) is total EPS motor output — NOT driver input.
    ret.steeringTorque = cp.vl["STEER_MODULE_2"]["DRIVER_EPS_TORQUE"]
    ret.steeringTorqueEps = cp.vl["STEERING_TORQUE"]["MAIN_TORQUE"]
    ret.steeringPressed = ret.steeringTorque > 80  # raw threshold; observed max ~52 during normal turns

    # --- Pedals ---
    ret.gasPressed = cp.vl["PEDAL"]["GAS_PEDAL"] > 1.0
    brake_pedal = cp.vl["PEDAL"]["BRAKE_PEDAL"]  # physical, DBC factor 0.01 already applied
    # BRAKE_PEDAL is the only signal that tracks the driver's foot. Verified against a
    # controlled pedal capture and against 3.6h of driving:
    #   - DRIVE_STATE.BRAKE_PRESSED (DBC bit 37) is dead — byte 4 is a constant 0x0C.
    #   - PEDAL_PRESSED_ACTIVE_LOW is the brake-light switch: 86% of its assertions on
    #     the road happened while the camera's ACC was commanding decel, not the driver.
    ret.brakePressed = brake_pedal > 0.03  # raw > 3, matches BYD_BRAKE_THRESHOLD in panda

    # --- Gear ---
    gear_map = {
      1: structs.CarState.GearShifter.park,
      2: structs.CarState.GearShifter.reverse,
      4: structs.CarState.GearShifter.drive,
    }
    ret.gearShifter = gear_map.get(int(cp.vl["DRIVE_STATE"]["GEAR"]),
                                   structs.CarState.GearShifter.unknown)

    # --- Wheel speeds ---
    # DBC factor 0.1 gives km/h, but BYD ATTO3 India raw values read ~32.5% high
    # vs odometer at steady-state. Correction: 40/53. Verify with GPS if re-calibrating.
    _SPD_CORR = 40.0 / 53.0
    fl = cp.vl["WHEEL_SPEED"]["WHEELSPEED_FL"] * _SPD_CORR / 3.6
    fr = cp.vl["WHEEL_SPEED"]["WHEELSPEED_FR"] * _SPD_CORR / 3.6
    rl = cp.vl["WHEEL_SPEED"]["WHEELSPEED_BL"] * _SPD_CORR / 3.6
    # WHEELSPEED_BR (bits 48-63): byte 7 is a constant status byte (0x41),
    # NOT the high byte of the wheel speed. DBC wrongly declares it as 16-bit.
    # Until the DBC is corrected, derive RR from the other three wheels.
    rr = (fl + fr + rl) / 3.0
    ret.wheelSpeeds.fl = fl
    ret.wheelSpeeds.fr = fr
    ret.wheelSpeeds.rl = rl
    ret.wheelSpeeds.rr = rr
    ret.vEgoRaw = (fl + fr + rl) / 3.0
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.05

    # --- Cruise / ACC --- (read from camera bus 2 — native source)
    # ACC_HUD_ADAS.ACC_ON1/ACC_ON2 are the ACC *main switch / standby* state, NOT
    # engagement. Proven on the 2026-08-07 drive: 25 min of stop-and-go with the driver
    # braking 28% of the time and both bits set throughout. The real engaged flag is in
    # ACC_CMD: CMD_REQ_ACTIVE_LOW == 0 (with ACC_ON_1/2 set) only while the stock ACC is
    # actively commanding. Using standby as "enabled" trapped openpilot after every brake
    # tap: cruiseState.enabled never dropped, so pcmEnable never saw a rising edge again.
    acc_main = bool(cp_cam.vl["ACC_HUD_ADAS"]["ACC_ON1"]) or bool(cp_cam.vl["ACC_HUD_ADAS"]["ACC_ON2"])
    acc_engaged = (bool(cp_cam.vl["ACC_CMD"]["ACC_ON_1"]) and bool(cp_cam.vl["ACC_CMD"]["ACC_ON_2"])
                   and not bool(cp_cam.vl["ACC_CMD"]["CMD_REQ_ACTIVE_LOW"]))

    # Debounce the drop: a single corrupted frame must not disengage.
    if acc_engaged:
      self.acc_off_frames = 0
    else:
      self.acc_off_frames += 1
    if self.acc_off_frames == 0:
      self.cruise_enabled_last = True
    elif self.acc_off_frames >= CRUISE_DROP_FRAMES:
      self.cruise_enabled_last = False

    ret.cruiseState.enabled = self.cruise_enabled_last
    # available is the ACC main switch, not the engaged state — aliasing the two made
    # every disengage also raise wrongCarMode and block re-engagement.
    ret.cruiseState.available = acc_main
    # DBC SET_SPEED factor 0.5 already applied (gives km/h); convert to m/s
    ret.cruiseState.speed = cp_cam.vl["ACC_HUD_ADAS"]["SET_SPEED"] / 3.6
    ret.cruiseState.standstill = False

    # We take over LKAS entirely: panda blocks the camera's steering command and we
    # send our own, so the stock system is never a competing controller.
    ret.stockLkas = False
    self.lkas_hud = {k: cp_cam.vl["LKAS_HUD_ADAS"][k] for k in LKAS_HUD_PASSTHROUGH}

    self.update_steer_template(cp_cam.vl["STEERING_MODULE_ADAS"])

    # --- Safety ---
    ret.seatbeltUnlatched = not bool(cp.vl["METER_CLUSTER"]["SEATBELT_DRIVER"])
    ret.doorOpen = any([
      cp.vl["METER_CLUSTER"]["FRONT_LEFT_DOOR"],
      cp.vl["METER_CLUSTER"]["FRONT_RIGHT_DOOR"],
      cp.vl["METER_CLUSTER"]["BACK_LEFT_DOOR"],
      cp.vl["METER_CLUSTER"]["BACK_RIGHT_DOOR"],
    ])

    # --- Blinkers ---
    ret.leftBlinker = bool(cp.vl["STALKS"]["LEFT_BLINKER"])
    ret.rightBlinker = bool(cp.vl["STALKS"]["RIGHT_BLINKER"])

    # --- Button events ---
    button_events = []
    for button in BUTTONS:
      if button.can_addr not in ("PCM_BUTTONS", "STALKS"):
        continue
      msg_val = bool(cp.vl[button.can_addr][button.can_msg])
      if msg_val != self.button_states[button.event_type]:
        event = structs.CarState.ButtonEvent.new_message()
        event.type = button.event_type
        event.pressed = msg_val
        button_events.append(event)
        self.button_states[button.event_type] = msg_val
    ret.buttonEvents = button_events

    ret_sp = structs.CarStateSP()
    return ret, ret_sp

  def _update_song(self, can_parsers) -> tuple[structs.CarState, structs.CarStateSP]:
    """SONG PLUS DMI state (ported from the opendbc-master reference).

    The stock camera (MPC) is the master: ACC_MPC_STATE carries the LKAS torque the
    camera wants, ACC_HUD_ADAS carries the cruise state, and the EPS answers on
    ACC_EPS_STATE. openpilot mirrors all three (plus ACC_CMD and RADAR_MRR) so it can
    impersonate either side later.
    """
    cp = can_parsers[Bus.pt]       # bus 0: ESC / powertrain CAN
    cp_cam = can_parsers[Bus.cam]  # bus 2: camera (MPC) CAN
    ret = structs.CarState.new_message()
    ret_sp = structs.CarStateSP()

    self.lkas_prepared = cp.vl["ACC_EPS_STATE"]["LKAS_Prepared"]

    self.mpc_lkas_config = int(cp_cam.vl["ACC_MPC_STATE"]["LKAS_Config"])
    lkas_config_isAccOn = (self.mpc_lkas_config != LKASConfig.DISABLE)
    lkas_isMainSwOn = bool(cp.vl["PCM_BUTTONS"]["BTN_TOGGLE_ACC_OnOff"])

    lkas_hud_AccOn1 = bool(cp_cam.vl["ACC_HUD_ADAS"]["AccOn1"])
    self.acc_state = cp_cam.vl["ACC_HUD_ADAS"]["AccState"]
    self.adas_set_dist = cp_cam.vl["ACC_HUD_ADAS"]["SetDistance"]

    prev_btn_acc_cancel = self.btn_acc_cancel
    prev_btn_acc_set_reset = self.btn_acc_set_reset
    prev_btn_acc_dist_inc = self.btn_acc_dist_inc
    prev_btn_acc_dist_dec = self.btn_acc_dist_dec

    self.btn_acc_cancel = cp.vl["PCM_BUTTONS"]["BTN_AccCancel"]
    self.btn_acc_set_reset = cp.vl["PCM_BUTTONS"]["BTN_AccUpDown_Cmd"]
    self.btn_acc_dist_inc = cp.vl["PCM_BUTTONS"]["BTN_AccDistanceIncrease"]
    self.btn_acc_dist_dec = cp.vl["PCM_BUTTONS"]["BTN_AccDistanceDecrease"]

    # use dash speedo as speed reference
    speed_raw = int(cp.vl["CARSPEED"]["CarDisplaySpeed"])
    speed_raw_kph = speed_raw * SongCarControllerParams.K_DASHSPEED
    correct_factor = np.interp(speed_raw_kph, [30, 60, 90, 120], [1., 1., 1., 1.])
    self.speed_kph = speed_raw_kph * correct_factor

    ret.vEgoRaw = float(self.speed_kph * CV.KPH_TO_MS)  # KPH to m/s
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.standstill = (speed_raw == 0)

    if self.CP.minSteerSpeed > 0:
      if self.speed_kph > 0.5:
        self.lkas_allowed_speed = True
      elif self.speed_kph < 0.1:
        self.lkas_allowed_speed = False
    else:
      self.lkas_allowed_speed = True

    can_gear = int(cp.vl["DRIVE_STATE"]["Gear"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.genericToggle = bool(cp.vl["STALKS"]["HeadLight"])
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(cp.vl["BSD_RADAR"]["LEFT_APPROACH"])
      ret.rightBlindspot = bool(cp.vl["BSD_RADAR"]["RIGHT_APPROACH"])

    ret.leftBlinker = bool(cp.vl["STALKS"]["LeftIndicator"])
    ret.rightBlinker = bool(cp.vl["STALKS"]["RightIndicator"])

    ret.steeringAngleOffsetDeg = 0
    ret.steeringAngleDeg = cp.vl["EPS"]["SteeringAngle"]

    self.steeringRateDegAbs = cp.vl["EPS"]["SteeringAngleRate"]
    ret.steeringRateDeg = self.steeringRateDegAbs

    ret.steeringTorque = cp.vl["ACC_EPS_STATE"]["SteerDriverTorque"]
    ret.steeringTorqueEps = cp.vl["ACC_EPS_STATE"]["MainTorque"]
    self.eps_warning = bool(cp.vl["ACC_EPS_STATE"]["SteerWarning"])  # todo: some firmware have SteerWarning field asserted
    self.eps_state_counter = int(cp.vl["ACC_EPS_STATE"]["Counter"])

    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > 59, 5)

    ret.parkingBrake = (cp.vl["EPB"]["EPB_ActiveFlag"] == 1)

    ret.brakePressed = (int(cp.vl["PEDAL"]["BrakePedal"]) != 0)

    ret.seatbeltUnlatched = (cp.vl["BCM"]["DriverSeatBeltFasten"] != 1)

    ret.doorOpen = any([cp.vl["BCM"]["FrontLeftDoor"], cp.vl["BCM"]["FrontRightDoor"],
                        cp.vl["BCM"]["RearLeftDoor"], cp.vl["BCM"]["RearRightDoor"]])

    ret.gasPressed = (int(cp.vl["PEDAL"]["AcceleratorPedal"]) != 0)

    ret.cruiseState.available = lkas_isMainSwOn and lkas_config_isAccOn and lkas_hud_AccOn1
    ret.cruiseState.enabled = self.acc_state in (3, 5)
    ret.cruiseState.standstill = ret.standstill
    ret.cruiseState.speed = cp_cam.vl["ACC_HUD_ADAS"]["SetSpeed"] * CV.KPH_TO_MS

    # todo: some firmware have these fields asserted
    ret.steerFaultTemporary = bool((self.acc_state == 7) or self.eps_warning)

    self.acc_active_last = ret.cruiseState.enabled

    # use to fool the MPC (see create_fake_318)
    self.mpc_laks_output = cp_cam.vl["ACC_MPC_STATE"]["LKAS_Output"]
    self.mpc_laks_reqprepare = cp_cam.vl["ACC_MPC_STATE"]["LKAS_ReqPrepare"] != 0
    self.mpc_laks_active = cp_cam.vl["ACC_MPC_STATE"]["LKAS_Active"] != 0

    self.acc_hud_adas_counter = cp_cam.vl["ACC_HUD_ADAS"]["Counter"]
    self.acc_mpc_state_counter = cp_cam.vl["ACC_MPC_STATE"]["Counter"]
    self.acc_cmd_counter = cp_cam.vl["ACC_CMD"]["Counter"]

    # whole-frame copies so the controller can replay the camera's exact frames
    self.cam_lkas = copy.copy(cp_cam.vl["ACC_MPC_STATE"])
    self.cam_adas = copy.copy(cp_cam.vl["ACC_HUD_ADAS"])
    self.cam_acc = copy.copy(cp_cam.vl["ACC_CMD"])
    self.esc_eps = copy.copy(cp.vl["ACC_EPS_STATE"])

    if not self.CP.radarUnavailable:
      mrr_id = int(cp_cam.vl["RADAR_MRR"]["TargetID"])
      if mrr_id == 2:  # 1:left, 2:front, 3:right
        if bool(cp_cam.vl["RADAR_MRR"]["IsValid"]):
          self.mrr_leading_dist = int(cp_cam.vl["RADAR_MRR"]["LongDist"])
        else:
          self.mrr_leading_dist = 199

    ret.steerFaultPermanent = bool(cp.vl["ACC_EPS_STATE"]["TorqueFailed"])  # EPS gives up all inputs until restart

    ret.buttonEvents = [
      *create_button_events(self.btn_acc_cancel, prev_btn_acc_cancel, {1: ButtonType.cancel}),
      *create_button_events(self.btn_acc_set_reset, prev_btn_acc_set_reset, {1: ButtonType.decelCruise, 3: ButtonType.accelCruise}),
      *create_button_events(self.btn_acc_dist_inc, prev_btn_acc_dist_inc, {1: ButtonType.gapAdjustCruise}),
      *create_button_events(self.btn_acc_dist_dec, prev_btn_acc_dist_dec, {1: ButtonType.gapAdjustCruise}),
    ]

    return ret, ret_sp

  @staticmethod
  def get_can_parsers(CP, CP_SP):
    if CP.carFingerprint in PLATFORM_SONG_PLUS_DMI:
      # SONG PLUS DMI (opendbc-master reference): both parsers read byd_song.dbc, one
      # on the ESC bus and one on the camera (MPC) bus.
      pt_messages = [
        # sig_address, frequency
        ("EPS", 100),
        ("CARSPEED", 50),
        ("PEDAL", 50),
        ("EPB", 1),
        ("ACC_EPS_STATE", 50),
        ("DRIVE_STATE", 50),
        ("STALKS", 1),
        ("BCM", 1),
        ("PCM_BUTTONS", 20),
        ("DATETIME", 2),
      ]
      if CP.enableBsm:
        pt_messages.append(("BSD_RADAR", 20))

      cam_messages = [
        ("ACC_HUD_ADAS", 50),
        ("ACC_CMD", 50),
        ("ACC_MPC_STATE", 50),
      ]
      if not CP.radarUnavailable:
        cam_messages.append(("RADAR_MRR", 60))

      return {
        Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, CanBus.ESC),
        Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, CanBus.MPC),
      }

    # Bus 0: messages sent by car ECUs (EPS, BCM, wheel speed, pedals, etc.)
    pt_messages = [
      ("STEER_MODULE_2", 0),
      ("STEERING_TORQUE", 0),
      ("PEDAL", 0),
      ("DRIVE_STATE", 0),
      ("WHEEL_SPEED", 0),
      ("METER_CLUSTER", 0),
      ("PCM_BUTTONS", 0),
      ("STALKS", 0),
    ]
    # Bus 2: messages sent by the ADAS camera module
    cam_messages = [
      ("ACC_HUD_ADAS", 0),
      ("ACC_CMD", 0),
      ("LKAS_HUD_ADAS", 0),
      ("STEERING_MODULE_ADAS", 0),
    ]
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.cam], cam_messages, 2),
    }
