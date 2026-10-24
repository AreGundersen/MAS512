
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Paths (same layout as Q1_Camera.py)
HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "Q1" / "BoxCenterXYZ.csv"
OUTPUT_DIR = HERE / "Q1"

# Intrinsic matrix M_int from the lab: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  -> FILL IN
FOCAL_X_INT = None
FOCAL_Y_INT = None
PRINCIPAL_X_INT = None
PRINCIPAL_Y_INT = None

# Default intrinsic matrix M_d of the L515 (given in the assignment)
FOCAL_X_DEFAULT = 1342.0
FOCAL_Y_DEFAULT = 1342.0
PRINCIPAL_X_DEFAULT = 325.0
PRINCIPAL_Y_DEFAULT = 235.0

# Optional: the real, measured length of the pendulum string in meters (used as reference in (c))
MEASURED_PENDULUM_LENGTH_M = None

# Machine learning settings
TEST_FRACTION = 0.2
RANDOM_SEED = 42

if None in (FOCAL_X_INT, FOCAL_Y_INT, PRINCIPAL_X_INT, PRINCIPAL_Y_INT):
    raise SystemExit("Fill in M_int (FOCAL_X_INT, FOCAL_Y_INT, PRINCIPAL_X_INT, PRINCIPAL_Y_INT) at the top of the file.")


def pixelToCamera(pixelU, pixelV, depthMeters, focalX, focalY, principalX, principalY):
    """Pinhole back-projection: pixel (u, v) with depth Z [m] -> camera-frame point (X, Y, Z) [m]."""
    cameraX = (pixelU - principalX) * depthMeters / focalX
    cameraY = (pixelV - principalY) * depthMeters / focalY
    return np.column_stack([cameraX, cameraY, depthMeters])


def plotTrajectory3d(points, frames, title, filename):
    """3D plot of the load centre, coloured by frame number."""
    figure = plt.figure(figsize=(7, 6))
    axes = figure.add_subplot(projection="3d")
    # Camera frame: x right, y DOWN, z forward. We plot -y as the vertical axis so that y points up
    # as in the assignment figure, and z (depth) as the horizontal axis going into the scene.
    axes.plot(points[:, 0], points[:, 2], -points[:, 1], color="lightgray", linewidth=0.8)
    scatter = axes.scatter(points[:, 0], points[:, 2], -points[:, 1], c=frames, cmap="viridis", s=10)
    axes.set_xlabel("x [m]")
    axes.set_ylabel("z (depth) [m]")
    axes.set_zlabel("y (up) [m]")
    axes.set_title(title)
    figure.colorbar(scatter, label="frame", shrink=0.6)
    figure.savefig(OUTPUT_DIR / filename, dpi=150)


def circleFitResidual(points):
    """Fit a plane to the 3D points, then a circle inside that plane.
    A pendulum should trace a circular arc, so a small residual means a physically consistent trajectory.
    Returns (fitted radius [m], RMS residual [m])."""
    centered = points - points.mean(axis=0)
    _, _, rightSingularVectors = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ rightSingularVectors[:2].T       # 2D coordinates inside the best-fit plane
    planarX, planarY = planar[:, 0], planar[:, 1]
    # Algebraic circle fit: x^2 + y^2 = 2*cx*x + 2*cy*y + offset
    design = np.column_stack([2 * planarX, 2 * planarY, np.ones_like(planarX)])
    target = planarX ** 2 + planarY ** 2
    (centerX, centerY, offsetTerm), *_ = np.linalg.lstsq(design, target, rcond=None)
    radius = np.sqrt(offsetTerm + centerX ** 2 + centerY ** 2)
    residual = np.hypot(planarX - centerX, planarY - centerY) - radius
    return radius, np.sqrt(np.mean(residual ** 2))


def describeErrors(name, errorsMeters):
    """Print mean / median / std / max of a Euclidean error array in millimeters."""
    millimeters = errorsMeters * 1000
    print(f"{name}: mean={millimeters.mean():.1f} mm, median={np.median(millimeters):.1f} mm, "
          f"std={millimeters.std():.1f} mm, max={millimeters.max():.1f} mm")


# ---------- Load the box centres and depth from Q1_Camera.py ----------
data = np.loadtxt(INPUT_CSV, delimiter=",", skiprows=1)    # columns: frame, cx, cy, Z, searchRadius
validRows = ~np.isnan(data[:, 1]) & ~np.isnan(data[:, 3])  # drop frames without detection or depth
frames = data[validRows, 0]
pixelU = data[validRows, 1]
pixelV = data[validRows, 2]
depthMeters = data[validRows, 3]
print(f"Using {validRows.sum()} of {len(data)} frames")

# ---------- (a) and (b): 3D coordinates with the two intrinsic matrices ----------
o1 = pixelToCamera(pixelU, pixelV, depthMeters, FOCAL_X_INT, FOCAL_Y_INT, PRINCIPAL_X_INT, PRINCIPAL_Y_INT)
o2 = pixelToCamera(pixelU, pixelV, depthMeters, FOCAL_X_DEFAULT, FOCAL_Y_DEFAULT, PRINCIPAL_X_DEFAULT, PRINCIPAL_Y_DEFAULT)
plotTrajectory3d(o1, frames, "o1: swinging load with M_int", "o1_trajectory.png")
plotTrajectory3d(o2, frames, "o2: swinging load with M_d", "o2_trajectory.png")

# ---------- (c): which intrinsic matrix is more plausible? ----------
# Z comes directly from the depth sensor, so the matrices only change x and y.
# Check how well each trajectory fits a circular arc (pendulum) - lower residual = more consistent.
for name, points in (("o1 (M_int)", o1), ("o2 (M_d)", o2)):
    radius, rmsResidual = circleFitResidual(points)
    print(f"(c) {name}: fitted pendulum radius = {radius:.3f} m, circle-fit RMS residual = {rmsResidual * 1000:.1f} mm")
    if MEASURED_PENDULUM_LENGTH_M is not None:
        print(f"    difference to measured string length: {(radius - MEASURED_PENDULUM_LENGTH_M) * 1000:.1f} mm")

# ---------- (d): learn f_theta(o1) = o2* ----------
o1Train, o1Test, o2Train, o2Test = train_test_split(o1, o2, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
model = LinearRegression().fit(o1Train, o2Train)
o2Predicted = model.predict(o1Test)

print("\n(d) KPIs on the held-out test set:")
for axisIndex, axisName in enumerate("XYZ"):
    rmse = np.sqrt(mean_squared_error(o2Test[:, axisIndex], o2Predicted[:, axisIndex]))
    mae = mean_absolute_error(o2Test[:, axisIndex], o2Predicted[:, axisIndex])
    r2 = r2_score(o2Test[:, axisIndex], o2Predicted[:, axisIndex])
    print(f"    {axisName}: RMSE={rmse * 1000:.3f} mm, MAE={mae * 1000:.3f} mm, R2={r2:.6f}")

# Both matrices are pinhole models, so o2 is an exact linear function of o1:
#   X2 = (fx1/fx2) * X1 + ((cx1 - cx2)/fx2) * Z,   Y2 = (fy1/fy2) * Y1 + ((cy1 - cy2)/fy2) * Z,   Z2 = Z1
expectedWeights = np.array([
    [FOCAL_X_INT / FOCAL_X_DEFAULT, 0, (PRINCIPAL_X_INT - PRINCIPAL_X_DEFAULT) / FOCAL_X_DEFAULT],
    [0, FOCAL_Y_INT / FOCAL_Y_DEFAULT, (PRINCIPAL_Y_INT - PRINCIPAL_Y_DEFAULT) / FOCAL_Y_DEFAULT],
    [0, 0, 1],
])
np.set_printoptions(precision=4, suppress=True)
print("Learned weights (rows = X2, Y2, Z2; columns = X1, Y1, Z1):\n", model.coef_)
print("Analytical weights:\n", expectedWeights)

# ---------- (e): Euclidean errors ----------
o2Star = model.predict(o1)                                  # prediction for every valid frame
errorO1O2 = np.linalg.norm(o1 - o2, axis=1)                 # (i)  o1 vs o2
errorO1O2Star = np.linalg.norm(o1 - o2Star, axis=1)         # (ii) o1 vs o2*
errorModel = np.linalg.norm(o2 - o2Star, axis=1)            # extra: how well o2* reproduces o2
print("\n(e) Euclidean errors:")
describeErrors("(i)  |o1 - o2|  ", errorO1O2)
describeErrors("(ii) |o1 - o2*| ", errorO1O2Star)
describeErrors("     |o2 - o2*| ", errorModel)

figure, axes = plt.subplots(figsize=(8, 4))
axes.plot(frames, errorO1O2 * 1000, label="|o1 - o2|")
axes.plot(frames, errorO1O2Star * 1000, "--", label="|o1 - o2*|")
axes.plot(frames, errorModel * 1000, label="|o2 - o2*|")
axes.set_xlabel("frame")
axes.set_ylabel("Euclidean error [mm]")
axes.legend()
figure.savefig(OUTPUT_DIR / "euclidean_errors.png", dpi=150)

plt.show()