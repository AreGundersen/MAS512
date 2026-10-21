import math
from collections import deque

from controller import Supervisor

# params
R_W = 0.0625        # drive wheel radius [m]
B = 0.2529          # wheel width [m]
LIDAR_X = 0.19731   
ARENA = 4.95  
V_MIN = 0.05        # minimum speed [m/s]      
V_MAX = 0.8         # max speed [m/s]
ACC = 3.0           # 
STOP_DIST = 0.6     # stop when lidar is ths far from goal/target [m]
CORRIDOR = 0.5      # 
FRONT = 0.32        # how forward from the origo the robot is [m]
D_CLEAR = 1.2       # clear distance in front of the robot [m]
WIN = 10            # 
T_MAX = 120.0       # tidsavbrudd [s]
BODY_X = 0.26       # lidar: avoid own chassis(x) + small margine
BODY_Y = 0.38       # lidar: avoid own chassis(y) + small margine
INF = float('inf')

# startup
robot = Supervisor()
STEP = int(robot.getBasicTimeStep())
DT = STEP / 1000.0

motor_l = robot.getDevice("left_wheel_motor")
motor_r = robot.getDevice("right_wheel_motor")
for m in (motor_l, motor_r):
    m.setPosition(INF)
    m.setVelocity(0.0)
W_LIM = motor_l.getMaxVelocity()

enc_l = robot.getDevice("left_wheel_sensor")
enc_r = robot.getDevice("right_wheel_sensor")
enc_l.enable(STEP)
enc_r.enable(STEP)

lidar = robot.getDevice("lidar")
lidar.enable(4 * STEP)
N = lidar.getHorizontalResolution()
L_MAX = lidar.getMaxRange()
ANG = None          

node = robot.getSelf()


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# lidar 
def calibrate(ranges):
    p, Rm = node.getPosition(), node.getOrientation()
    yaw = math.atan2(Rm[3], Rm[0])
    lx, ly = p[0] + LIDAR_X * math.cos(yaw), p[1] + LIDAR_X * math.sin(yaw)

    def wall(a):
        c, s = math.cos(a), math.sin(a)
        ts = []
        if abs(c) > 1e-9:
            ts.append(((ARENA if c > 0 else -ARENA) - lx) / c)
        if abs(s) > 1e-9:
            ts.append(((ARENA if s > 0 else -ARENA) - ly) / s)
        return min(ts)

    best = None
    for sign in (1, -1):
        for off in (0, math.pi / 2, math.pi, -math.pi / 2):
            err = sorted(abs(ranges[i] - wall(yaw + off + sign * i * 2 * math.pi / N))
                         for i in range(0, N, 5) if math.isfinite(ranges[i]) and ranges[i] > 0.6)
            score = err[len(err) // 2]
            if best is None or score < best[0]:
                best = (score, sign, off)
    print(f"Lidar-kalibrering: sign={best[1]:+d}, offset={math.degrees(best[2]):.0f} deg, "
          f"feil={best[0]:.3f} m")
    return [best[2] + best[1] * i * 2 * math.pi / N for i in range(N)]


def find_goal(rg):
    r = [v if (math.isfinite(v) and v < 0.98 * L_MAX) else INF for v in rg]
    jump = lambda a, b: a == INF or b == INF or abs(a - b) > 0.15 + 0.03 * min(a, b)
    start = next((i for i in range(N) if jump(r[i - 1], r[i])), None)
    if start is None:
        return None
    found, cl = [], []
    for k in range(N + 1):
        i = (start + k) % N
        if k < N and r[i] != INF and (not cl or not jump(r[cl[-1]], r[i])):
            cl.append(i)
            continue
        if len(cl) >= 3:
            a, b = cl[0], cl[-1]
            width = math.hypot(r[a] * math.cos(ANG[a]) - r[b] * math.cos(ANG[b]),
                               r[a] * math.sin(ANG[a]) - r[b] * math.sin(ANG[b]))
            rmin = min(r[j] for j in cl)
            na, nb = r[a - 1], r[(b + 1) % N]                      
            alone = na != INF and nb != INF and na >= rmin + 0.12 and nb >= rmin + 0.12
            if 0.08 <= width <= 0.30 and alone:
                m = cl[len(cl) // 2]
                x, y = LIDAR_X + r[m] * math.cos(ANG[m]), r[m] * math.sin(ANG[m])
                found.append((math.atan2(y, x), rmin, x, y))
        cl = [i] if (k < N and r[i] != INF) else []
    return min(found, key=lambda g: g[1]) if found else None


def free_dist(theta, obst):
    c, s = math.cos(theta), math.sin(theta)
    d = 3.0
    for x, y in obst:
        f = x * c + y * s
        if 0 < f < d and abs(-x * s + y * c) < CORRIDOR:
            d = f
    return max(0.0, d - FRONT)


# main loop
state, goal, goal_t = "SEARCH", None, -1e9
stop_t = None
v_cmd = 0.0
v_tgt = w_tgt = 0.0
done = False
th_l, th_r = deque(maxlen=WIN + 1), deque(maxlen=WIN + 1)
vx_hist = deque(maxlen=WIN)
step = 0

while robot.step(STEP) != -1:
    t = robot.getTime()
    step += 1
    th_l.append(enc_l.getValue())
    th_r.append(enc_r.getValue())

    # true speed at x axis (supervisor)
    vel, Rm, pos = node.getVelocity(), node.getOrientation(), node.getPosition()
    vx_hist.append(vel[0] * Rm[0] + vel[1] * Rm[3] + vel[2] * Rm[6])

    if step % 4 == 0 and t >= 0.1 and not done:
        rg = lidar.getRangeImage()
        if ANG is None:
            ANG = calibrate(rg)
        # ignore own chassis (backside of lidar)
        rg = [INF if (math.isfinite(v) and abs(LIDAR_X + v * math.cos(ANG[i])) < BODY_X
                      and abs(v * math.sin(ANG[i])) < BODY_Y) else v for i, v in enumerate(rg)]
        g = find_goal(rg)
        if g:
            goal, goal_t = g, t
        known = goal is not None and t - goal_t < 1.0     

        pts = [(LIDAR_X + rg[i] * math.cos(ANG[i]), rg[i] * math.sin(ANG[i]))
               for i in range(0, N, 3) if 0.1 < rg[i] < 3.0]
        if known:                                         
            pts = [p for p in pts if math.hypot(p[0] - goal[2], p[1] - goal[3]) > 0.35]

        if state == "SEARCH" and known:
            state = "APPROACH"
        elif state == "APPROACH" and not known:
            state = "SEARCH"
        if state == "APPROACH" and goal[1] < STOP_DIST and t - goal_t < 0.1:
            state, stop_t = "STOP", t

        # choose direction
        target = goal[0] if state == "APPROACH" else 0.0
        theta = min((math.radians(d) for d in range(-180, 181, 6)),
                    key=lambda a: abs(wrap(a - target)) + 2.0 * max(0.0, 1 - free_dist(a, pts) / D_CLEAR))
        free = free_dist(theta, pts)

        v_tgt = V_MAX * clamp((free - 0.1) / D_CLEAR, 0, 1)
        v_tgt *= max(0.0, math.cos(theta)) ** 2 if abs(theta) < 1.2 else 0.0   
        if state == "APPROACH":
            v_tgt = min(v_tgt, max(0.1, math.sqrt(2 * max(0.0, goal[1] - STOP_DIST))))   # break when close to target
        w_tgt = clamp(2.0 * theta, -2.0, 2.0)

    if state == "STOP" or done:
        v_tgt = w_tgt = 0.0

    # helping commands
    v_cmd += clamp(v_tgt - v_cmd, -ACC * DT, ACC * DT)
    motor_l.setVelocity(clamp((v_cmd - w_tgt * B / 2) / R_W, -W_LIM, W_LIM))
    motor_r.setVelocity(clamp((v_cmd + w_tgt * B / 2) / R_W, -W_LIM, W_LIM))

    # slip: s = (v_enc - v_x) / v_x 
    nan = float('nan')
    v_enc = v_x = s = nan
    if len(th_l) == WIN + 1:
        v_enc = R_W * ((th_l[-1] - th_l[0]) + (th_r[-1] - th_r[0])) / (2 * WIN * DT)
        v_x = sum(vx_hist) / WIN
        if abs(v_x) > V_MIN:
            s = (v_enc - v_x) / v_x

    if step % 125 == 0:
        print(f"t={t:5.1f} {state:8s} pos=({pos[0]:5.2f},{pos[1]:5.2f}) v_enc={v_enc:6.3f} v_x={v_x:6.3f} s={s:7.3f}")

    # FINISH
    if not done and ((state == "STOP" and t - stop_t > 1.5) or t > T_MAX):
        done = True
        print("Ferdig.")
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)