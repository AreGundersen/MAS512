"""slip_controller - enkel kontroller for Rosey (differensialdrift).

Roboten kjorer rett frem i noen sekunder, svinger pa stedet,
og kjorer rett frem igjen. Underveis skrives tilstand og
hjulvinkler fra encoderne til konsollen.
"""

from controller import Robot

# ---------------------------------------------------------------
# 1) Oppstart
# ---------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Rosey sine mal (fra MAS500 / CAD)
WHEEL_RADIUS = 0.0625   # m  (O125 mm)
TRACK_WIDTH = 0.2529    # m  (avstand mellom drivhjulene)

# ---------------------------------------------------------------
# 2) Skriv ut alle enheter roboten har
# ---------------------------------------------------------------
print("Enheter pa roboten:")
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    print(f"  - {device.getName()}")

# ---------------------------------------------------------------
# 3) Hent motorer og encodere
# ---------------------------------------------------------------
# Navnene ma matche 'name'-feltet i scenetreet.
left_motor = robot.getDevice("left_wheel_motor")
right_motor = robot.getDevice("right_wheel_motor")

# setPosition(inf) gir ren hastighetsstyring via setVelocity().
for m in (left_motor, right_motor):
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

left_encoder = robot.getDevice("left_wheel_sensor")
right_encoder = robot.getDevice("right_wheel_sensor")
left_encoder.enable(timestep)
right_encoder.enable(timestep)

MAX_SPEED = left_motor.getMaxVelocity()
print(f"Maks hjulhastighet: {MAX_SPEED:.2f} rad/s")


def set_wheel_speeds(left, right):
    """Setter hjulhastighet (rad/s) pa venstre og hoyre drivhjul.
      - left == right  -> rett frem/bakover
      - left == -right -> svinger pa stedet
    """
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)


# ---------------------------------------------------------------
# 4) Hovedlokke
# ---------------------------------------------------------------
SPEED = 2.0  # rad/s -> 2.0 * 0.0625 = 0.125 m/s

while robot.step(timestep) != -1:
    t = robot.getTime()

    if t < 5.0:
        set_wheel_speeds(SPEED, SPEED)
        state = "rett frem"
    elif t < 7.0:
        set_wheel_speeds(-SPEED, SPEED)
        state = "svinger"
    elif t < 12.0:
        set_wheel_speeds(SPEED, SPEED)
        state = "rett frem"
    else:
        set_wheel_speeds(0.0, 0.0)
        state = "stopp"

    # Skriv status ca. en gang i sekundet
    if int(t * 1000) % 1000 < timestep:
        theta_l = left_encoder.getValue()
        theta_r = right_encoder.getValue()
        print(f"t = {t:5.1f} s | {state:9s} | "
              f"venstre: {theta_l:7.2f} rad | hoyre: {theta_r:7.2f} rad")