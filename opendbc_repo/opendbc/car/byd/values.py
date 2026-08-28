from collections import namedtuple
from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car import Bus, CarSpecs, PlatformConfig, Platforms, structs
from opendbc.car.lateral import AngleSteeringLimits
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries
from opendbc.car.docs_definitions import CarDocs, CarParts, CarHarness, SupportType

# HUD multiplier for display calibration
HUD_MULTIPLIER = 0.718

Ecu = structs.CarParams.Ecu
ButtonType = structs.CarState.ButtonEvent.Type
Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])


# BYD Car Controller Parameters - tuned for ATTO3
class CarControllerParams:
  # STEERING_MODULE_ADAS.STEER_ANGLE is an *absolute steering wheel angle* target
  # (DBC factor 0.1 deg, same units as the STEER_ANGLE_2 sensor). The EPS is a
  # position servo, so the command must always stay anchored to the measured angle.
  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    # The EPS faults out past ~90 deg of commanded wheel angle.
    90.,  # deg
    # deg of change allowed per CAN message (50 Hz). Generous at parking speeds,
    # tightened at road speed to stay near ISO 11270 lateral accel/jerk.
    # 2.5 deg/msg @ 50 Hz = 125 deg/s; 0.4 deg/msg = 20 deg/s.
    ([0., 5., 25.], [2.5, 1.5, 0.4]),
    ([0., 5., 25.], [2.5, 1.5, 0.6]),
  )

  # Hard windup guard: the command may never sit further than this from the measured
  # angle. This is what makes a ratcheting/saturating limiter structurally impossible.
  MAX_ANGLE_ERROR = 12.             # deg

  # Driver override threshold on DRIVER_EPS_TORQUE (column sensor, raw 0-255).
  # Observed max ~52 during normal driver turns.
  STEER_DRIVER_ALLOWANCE = 80

  # Control timing - CAN messages sent at 50Hz (controller runs at 100Hz)
  STEER_STEP = 2

  ACCEL_MAX = 2.0                   # m/s^2
  ACCEL_MIN = -3.5                  # m/s^2

  def __init__(self, CP):
    pass


# SONG PLUS DMI / DMI-family parameters, ported from the opendbc-master reference
# implementation. Unlike the ATTO3 (angle-based EPS), this family is torque-based:
# openpilot impersonates the camera and drives 0x316 ACC_MPC_STATE.LKAS_Output on the
# ESC bus, while sending a fake 0x318 ACC_EPS_STATE back to the camera bus so the
# stock MPC keeps AEB etc. alive (no DTC).
class SongCarControllerParams(CarControllerParams):
  STEER_MAX = 300
  STEER_DELTA_UP = 7
  STEER_DELTA_DOWN = 10

  STEER_DRIVER_ALLOWANCE = 68
  STEER_DRIVER_MULTIPLIER = 3
  STEER_DRIVER_FACTOR = 1
  STEER_ERROR_MAX = 50

  STEER_STEP = 2  # 100/2 = 50 Hz
  STEER_SOFTSTART_STEP = 6  # 20ms(50Hz) * 300 / 6 = 1000ms until the clip ceiling reaches STEER_MAX

  ACC_STEP = 2  # 50 Hz

  ACCEL_MAX = 2.0
  ACCEL_MIN = -3.5

  K_DASHSPEED = 0.0719088  # convert dash pulse to kph

  USE_STEERING_SPEED_LIMITER = False

  # op long control jerk shaping
  K_accel_jerk_upper = 0.1
  K_accel_jerk_lower = 0.5
  K_jerk_xp = [4, 10, 20, 40, 80]  # meters
  K_jerk_base_lower_fp = [-2.3, -1.8, -1.4, -1.0, -0.4]
  K_jerk_base_upper_fp = [0.8, 0.7, 0.6, 0.3, 0.2]


# safetyParam flags distinguishing the BYD platforms (SAFETY_BYD shares one mode)
class BydSafetyFlags(IntFlag):
  HAN_TANG_DMEV = 0x1  # pre-2021 models with veoneer mpc/radar solution
  TANG_DMI = 0x2
  SONG_PLUS_DMI = 0x4  # note song pro is similar but not song dmi
  QIN_PLUS_DMI = 0x8
  YUAN_PLUS_DMI_ATTO3 = 0x10  # yuan plus is atto3


@dataclass
class BYDCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=lambda: CarParts.common([CarHarness.custom]))


@dataclass
class BYDPlatformConfig(PlatformConfig):
  dbc_dict: dict = field(default_factory=lambda: {
    Bus.pt: 'byd_general',   # bus 0: car-side chassis CAN
    Bus.cam: 'byd_general',  # bus 2: camera-side chassis CAN
  })


# BYD Vehicle Models
class CAR(Platforms):
  BYD_ATTO3 = BYDPlatformConfig(
    [
      BYDCarDocs("BYD ATTO3 2022-24", support_type=SupportType.COMMUNITY)
    ],
    CarSpecs(
      mass=1750,               # Vehicle mass in kg
      wheelbase=2.72,          # Wheelbase in meters
      steerRatio=14.8,         # Steering ratio
      tireStiffnessFactor=0.7983  # Tire stiffness factor
    ),
  )

  # SONG PLUS DMI family — byd_song.dbc, torque-based lateral (see SongCarControllerParams)
  BYD_SONG_PLUS_DMI_21 = BYDPlatformConfig(
    [BYDCarDocs("BYD SONG PLUS DMI 21")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    dbc_dict={Bus.pt: 'byd_song'},
  )

  BYD_SONG_PLUS_DMI_22 = BYDPlatformConfig(
    [BYDCarDocs("BYD SONG PLUS DMI 22")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    dbc_dict={Bus.pt: 'byd_song'},
  )

  BYD_SONG_PLUS_DMI_23 = BYDPlatformConfig(
    [BYDCarDocs("BYD SONG PLUS DMI 23")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    dbc_dict={Bus.pt: 'byd_song'},
  )

  BYD_SONG_PRO_DMI_22 = BYDPlatformConfig(
    [BYDCarDocs("BYD SONG PRO DMI 22")],
    CarSpecs(mass=1670., wheelbase=2.712, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    dbc_dict={Bus.pt: 'byd_song'},
  )


# CAN Bus Configuration
# Bus 0: Car-side chassis CAN (read car ECU signals from here)
# Bus 1: Private CAN (camera image/radar output, CAN-FD frames)
# Bus 2: Camera-side chassis CAN (camera sends ADAS messages here — send our overrides here)
class CanBus:
  pt = 0      # Car-side chassis CAN (receive)
  cam = 2     # Camera-side chassis CAN (send ADAS commands to compete with camera)
  radar = 1   # Private CAN (camera image/radar data)

  # SONG PLUS DMI family aliases (opendbc-master reference naming)
  ESC = 0     # ESC / powertrain bus
  MRR = 1     # radar bus
  MPC = 2     # camera (MPC) bus


# Every LKAS_HUD_ADAS field except the STEER_ACTIVE bits and the counter/checksum. openpilot
# only owns whether LKAS is shown as active; lane-line state, traffic sign recognition, high
# beam assist and the PT2-PT5 / SET_ME_* fields belong to the camera and blank out unrelated
# driver-assist icons if we zero them, so they are mirrored from its copy read on bus 2.
LKAS_HUD_PASSTHROUGH = ("LSS_STATE", "SETTINGS", "SET_ME_XFF", "SET_ME_X5F", "SET_ME_1_2",
                        "TSR", "HMA", "HAND_ON_WHEEL_WARNING", "PT2", "PT3", "PT4", "PT5")


# Button configurations for steering wheel controls
BUTTONS = [
  Button(ButtonType.leftBlinker, "STALKS", "LEFT_BLINKER", [1]),
  Button(ButtonType.rightBlinker, "STALKS", "RIGHT_BLINKER", [1]),
  Button(ButtonType.accelCruise, "PCM_BUTTONS", "RES_BTN", [1]),
  Button(ButtonType.decelCruise, "PCM_BUTTONS", "SET_BTN", [1]),
  Button(ButtonType.lkas, "PCM_BUTTONS", "LKAS_ON_BTN", [1]),
  Button(ButtonType.gapAdjustCruise, "PCM_BUTTONS", "DEC_DISTANCE_BTN", [1]),
  Button(ButtonType.gapAdjustCruise, "PCM_BUTTONS", "INC_DISTANCE_BTN", [1]),
]

# The ATTO3 fingerprints on its CAN message set (see fingerprints.py); the UDS query is here to
# collect firmware versions for a future FW fingerprint. Only the chassis bus is queried —
# bus 1 is the camera's private CAN and carries no diagnostic ECUs.
FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"[\x00-\xff]+",
  requests=[
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
  ],
  extra_ecus=[
    # the two diagnostic addresses that answered a UDS version request on the ATTO3
    (Ecu.engine, 0x7e0, None),
    (Ecu.hvac, 0x7b3, None),
  ],
)

class LKASConfig:
  DISABLE = 0
  ALARM = 1
  LKA = 2
  ALARM_AND_LKA = 3


PLATFORM_SONG_PLUS_DMI = {CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22, CAR.BYD_SONG_PLUS_DMI_23, CAR.BYD_SONG_PRO_DMI_22}

# DBC file mapping
DBC = CAR.create_dbc_map()
