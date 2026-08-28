from opendbc.car import get_safety_config, structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.byd.carstate import CarState
from opendbc.car.byd.carcontroller import CarController
from opendbc.car.byd.values import BydSafetyFlags, CanBus, PLATFORM_SONG_PLUS_DMI


NetworkLocation = structs.CarParams.NetworkLocation
TransmissionType = structs.CarParams.TransmissionType

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, alpha_long, is_release, docs):
    ret.brand = "byd"
    # SAFETY_BYD: relay engaged. openpilot owns STEERING_MODULE_ADAS and LKAS_HUD_ADAS on
    # bus 0; panda blocks the camera's copies from being forwarded 2->0 (check_relay) so the
    # EPS and cluster only ever see one source. All other camera messages pass through.
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.byd)]
    ret.radarUnavailable = True

    if candidate in PLATFORM_SONG_PLUS_DMI:
      # SONG PLUS DMI (opendbc-master reference): torque-based lateral. openpilot
      # impersonates the camera, driving ACC_MPC_STATE.LKAS_Output on the ESC bus while
      # faking ACC_EPS_STATE back to the camera bus (see bydcan.create_fake_318).
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.SONG_PLUS_DMI.value
      ret.enableBsm = 0x418 in fingerprint[CanBus.ESC]
      ret.transmissionType = TransmissionType.direct
      # the SONG family has a working front radar (RADAR_MRR), unlike the ATTO3
      ret.radarUnavailable = False

      ret.minSteerSpeed = 0.1 * CV.KPH_TO_MS
      ret.steerActuatorDelay = 0.05
      ret.steerLimitTimer = 0.4

      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[8.3, 27.8], [8.3, 27.8]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6, 0.3], [0.2, 0.1]]
      ret.lateralTuning.pid.kf = 0.000072

      # Longitudinal tuning is vestigial here (openpilotLongitudinalControl is False);
      # kpBP/kpV live in the deprecated group in this repo's car.capnp and are not set.
      ret.longitudinalTuning.kiBP = [0.]
      ret.longitudinalTuning.kiV = [0.3]

      # the reference marks every DMI platform dashcam-only until on-car verification
      ret.dashcamOnly = True
      ret.openpilotLongitudinalControl = False
      ret.pcmCruise = True
      ret.minEnableSpeed = -1
      return ret

    # BYD EPS receives absolute steering wheel angle targets in STEERING_MODULE_ADAS.
    ret.steerControlType = structs.CarParams.SteerControlType.angle
    # Measured on the car (2026-08-07 drive, cross-correlating the transmitted 0x1E2
    # angle against the 0x11F measured angle): the EPS follows with ~0.3-0.8 s of lag.
    # A higher value makes the planner start turning earlier into curves, which reduces
    # curve-entry steerSaturated events on this torque-limited EPS.
    ret.steerActuatorDelay = 0.35
    ret.steerLimitTimer = 0.4

    # Longitudinal tuning is vestigial here (openpilotLongitudinalControl is False);
    # kpBP/kpV live in the deprecated group in this repo's car.capnp and are not set.
    ret.longitudinalTuning.kiBP = [0., 35.]
    ret.longitudinalTuning.kiV = [0.18, 0.12]

    ret.networkLocation = NetworkLocation.fwdCamera

    ret.openpilotLongitudinalControl = False
    ret.pcmCruise = True
    ret.minEnableSpeed = -1
    ret.minSteerSpeed = 0.

    return ret
