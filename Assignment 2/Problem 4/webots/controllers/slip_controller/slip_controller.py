import math
import os
from controller import Supervisor

# parameters
R_W = 0.0625        # wheel radius [m]
B = 0.2529          # wheel track [m]
LIDAR_X = 0.19731   # lidar position along robot x [m]
V_MAX = 0.5         # top speed [m/s]
ACC = 1.5           # max speed change [m/s^2]
STOP_DIST = 0.6     # stop when lidar range to goal is below this [m]
CORRIDOR = 0.5      # half-width of free corridor [m]
LOOK = 1.6          # corridor length [m]
V_MIN = 0         
T_MAX = 120.0       # timeout [s]
INF = float('inf')

# setup 
robot = Supervisor()
node = robot.getSelf()
STEP = int(robot.getBasicTimeStep())
DT = STEP / 1000.0

motor_l = robot.getDevice("left_wheel_motor")
motor_r = robot.getDevice("right_wheel_motor")
motor_l.setPosition(INF)
motor_r.setPosition(INF)

enc_l = robot.getDevice("left_wheel_sensor")
enc_r = robot.getDevice("right_wheel_sensor")
enc_l.enable(STEP)
enc_r.enable(STEP)

lidar = robot.getDevice("lidar")
lidar.enable(4 * STEP)
N = lidar.getHorizontalResolution()

# logging (<webots>/results/<customData>.csv)
name = robot.getCustomData() or "run"
res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
os.makedirs(res_dir, exist_ok=True)
log = open(os.path.join(res_dir, name + ".csv"), "w")
log.write("t,state,x,y,v_cmd,v_enc,v_x,slip\n")
scan_log = open(os.path.join(res_dir, name + "_lidar.csv"), "w")
scan_log.write("t," + ",".join(f"r{i}" for i in range(N)) + "\n")


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def ray_angle(i):
    return math.pi - i * 2 * math.pi / N      # ray 0 points backwards, then clockwise


def find_goal(d):
    best = None
    i = 0
    while i < N:
        if d[i] == INF:
            i += 1
            continue
        j = i                                  # group neighbouring rays at similar range
        while j + 1 < N and abs(d[j + 1] - d[j]) < 0.15:
            j += 1
        near = min(d[i:j + 1])
        width = near * (j - i + 1) * 2 * math.pi / N
        a, b = d[i - 1], d[(j + 1) % N]                # both neighbours must be real hits behind it
        free_sides = near + 0.12 <= a < INF and near + 0.12 <= b < INF
        if j - i >= 2 and 0.08 < width < 0.30 and free_sides:
            if best is None or near < best[2]:
                m = (i + j) // 2
                best = (LIDAR_X + d[m] * math.cos(ray_angle(m)), d[m] * math.sin(ray_angle(m)), near)
        i = j + 1
    return best


def blocked(angle, pts):
    c, s = math.cos(angle), math.sin(angle)
    return any(0 < x * c + y * s < LOOK and abs(-x * s + y * c) < CORRIDOR for x, y in pts)


# --- main loop ---
state = "SEARCH"
goal, goal_t, stop_t = None, 0.0, 0.0
v_tgt = w_tgt = v_cmd = 0.0
th_l = th_r = 0.0                              # previous wheel angles
v_enc = v_x = 0.0                              # filtered speeds
slips = []
done = False
step = 0

while robot.step(STEP) != -1:
    if done:
        continue
    step += 1
    t = robot.getTime()

    # wheel speed from encoders
    new_l, new_r = enc_l.getValue(), enc_r.getValue()
    enc_speed = R_W * ((new_l - th_l) + (new_r - th_r)) / 2 / DT
    th_l, th_r = new_l, new_r

    # true forward speed from the supervisor (only used for slip)
    vel, Rm, pos = node.getVelocity(), node.getOrientation(), node.getPosition()
    true_speed = vel[0] * Rm[0] + vel[1] * Rm[3] + vel[2] * Rm[6]

    v_enc += 0.1 * (enc_speed - v_enc)
    v_x += 0.1 * (true_speed - v_x)
    slip = (v_enc - v_x) / v_x if abs(v_x) > V_MIN else float('nan')
    if slip == slip:
        slips.append(slip)

    # lidar
    if step % 4 == 0 and t > 0.1:
        r = lidar.getRangeImage()
        if step % 24 == 0:
            scan_log.write(f"{t:.3f}," + ",".join(f"{v:.3f}" for v in r) + "\n")

        # ranges -> points; ignore misses and hits on the robot's own body
        d, pts = [], []
        for i in range(N):
            x = LIDAR_X + r[i] * math.cos(ray_angle(i))
            y = r[i] * math.sin(ray_angle(i))
            own_body = abs(x) < 0.26 and abs(y) < 0.38
            if r[i] < 0.1 or r[i] > 19 or own_body:     # also catches inf
                d.append(INF)
            else:
                d.append(r[i])
                pts.append((x, y))

        found = find_goal(d)
        if found:
            goal, goal_t = found, t
        if t - goal_t > 1.0:                   # forget goal after 1 s without seeing it
            goal = None

        bearing = 0.0                          # no goal: drive straight
        if goal:
            pts = [p for p in pts if math.hypot(p[0] - goal[0], p[1] - goal[1]) > 0.35]   # goal is not an obstacle
            bearing = math.atan2(goal[1], goal[0])
        if state != "STOP":
            state = "APPROACH" if goal else "SEARCH"
            if found and found[2] < STOP_DIST:
                state, stop_t = "STOP", t

        # try headings next to the goal direction first
        options = [bearing + s * math.radians(15 * k) for k in range(13) for s in (1, -1)]
        heading = wrap(next((a for a in options if not blocked(a, pts)), math.pi))

        w_tgt = max(-2.0, min(2.0, 2.0 * heading))
        v_tgt = V_MAX * max(0.0, math.cos(heading)) ** 2   # slow down in turns
        if goal:
            v_tgt = min(v_tgt, 0.5 * (goal[2] - STOP_DIST) + 0.1)   # slow down near goal

    if state == "STOP":
        v_tgt = w_tgt = 0.0

    # ramp forward speed, then set wheel speeds
    v_cmd += max(-ACC * DT, min(ACC * DT, v_tgt - v_cmd))
    motor_l.setVelocity((v_cmd - w_tgt * B / 2) / R_W)
    motor_r.setVelocity((v_cmd + w_tgt * B / 2) / R_W)

    log.write(f"{t:.3f},{state},{pos[0]:.4f},{pos[1]:.4f},{v_cmd:.4f},{v_enc:.4f},{v_x:.4f},{slip:.5f}\n")
    if step % 125 == 0:
        print(f"t={t:5.1f} {state:8s} pos=({pos[0]:5.2f},{pos[1]:5.2f}) v_enc={v_enc:6.3f} v_x={v_x:6.3f} slip={slip:7.3f}")

    # FINISHED
    if (state == "STOP" and t - stop_t > 1.5) or t > T_MAX:
        motor_l.setVelocity(0.0)
        motor_r.setVelocity(0.0)
        log.close()
        scan_log.close()
        done = True
        if slips:
            print(f"Done. Slip mean {sum(slips) / len(slips):+.3f}, max {max(slips, key=abs):+.3f}")
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)