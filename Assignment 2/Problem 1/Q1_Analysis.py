from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Directories and paths (same layout as Q1_Camera.py)
HERE      = Path(__file__).resolve().parent
inputCsv  = HERE / "Q1_output" / "BoxCenterXYZ.csv"               # Box centre pixel and depth per frame, written by Q1_Camera.py
outputDir = HERE / "Q1_output"                                      # Plots are saved here

# The three intrinsic matrices that are compared, each [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# o1: own calibration (checkerboard) for the 640x480 video
focalXInt     = 661.795300
focalYInt     = 831.441419
principalXInt = 310.794541
principalYInt = 244.552865

# o3: intrinsic matrix fetched from the RealSense L515 (factory calibration, 640x480)
focalXCam     = 601.38623046875
focalYCam     = 601.5333862304688
principalXCam = 326.42962646484375
principalYCam = 239.318115234375

# o2: default intrinsic matrix M_d of the L515 (given in the assignment)
focalXDefault     = 1342.0
focalYDefault     = 1342.0
principalXDefault = 325.0
principalYDefault = 235.0

# Parameters
estimatedPendulumLength = 1.8        # Real length of the pendulum string in meters, used as reference in (c). None = not measured
testFraction           = 0.2        # Share of the frames that is held out for testing the model in (d)
randomSeed             = 42         # Fixed seed so that the split is the same every run
hiddenLayers           = (16, 16)   # Neural network for f_theta: two hidden layers with 16 neurons each
maxEpochs              = 300        # Maximum number of training epochs (passes over the training data)
stopLoss               = 1e-3       # Training stops early when the training loss (MSE of the scaled data) gets below this value. None = always run all epochs

# 1. Load the box centres and depth from Q1_Camera.py ---------------------------------------------------
data        = np.loadtxt(inputCsv, delimiter=",", skiprows=1)               # Columns: frame, cx, cy, Z, searchRadius
validRows   = ~np.isnan(data[:, 1]) & ~np.isnan(data[:, 3])                 # Dropping frames without detection or depth
frames      = data[validRows, 0]
pixelU      = data[validRows, 1]                                            # Box centre, horizontal pixel
pixelV      = data[validRows, 2]                                            # Box centre, vertical pixel
depthMeters = data[validRows, 3]                                            # Depth Z at the box centre in meters
print(f"Using {validRows.sum()} of {len(data)} frames")

# 2. (a) and (b): 3D coordinates with the two intrinsic matrices ----------------------------------------
# Pinhole back-projection: pixel (u, v) with depth Z -> camera-frame point  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy
intrinsics = {"o1 (M_int, own calibration)": (focalXInt, focalYInt, principalXInt, principalYInt),
              "o2 (M_d, given)":            (focalXDefault, focalYDefault, principalXDefault, principalYDefault),
              "o3 (M_cam, from the camera)": (focalXCam, focalYCam, principalXCam, principalYCam)}
trajectories = {}
for name, (fx, fy, cx, cy) in intrinsics.items():
    cameraX = (pixelU - cx) * depthMeters / fx                              # X to the right
    cameraY = (pixelV - cy) * depthMeters / fy                              # Y DOWN (camera convention)
    trajectories[name] = np.column_stack([cameraX, cameraY, depthMeters])   # (frames, 3), Z is the same for both matrices
o1, o2, o3 = trajectories.values()                                          # Same order as in the dictionary above

for (name, points), fileName in zip(trajectories.items(), ("o1_trajectory.png", "o2_trajectory.png", "o3_trajectory.png")):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    # Camera frame has y DOWN, so -y is plotted as the vertical axis (y up as in the assignment figure), z (depth) goes into the scene
    ax.plot(points[:, 0], points[:, 2], -points[:, 1], color="lightgray", linewidth=0.8)
    sc = ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=frames, cmap="viridis", s=10)   # Colour = frame number
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z (depth) [m]")
    ax.set_zlabel("y (up) [m]")
    ax.set_title(f"{name}: swinging load")
    fig.colorbar(sc, label="frame", shrink=0.6)
    fig.savefig(outputDir / fileName, dpi=150)

# All three trajectories in one figure: 3D overlay on the left, x, y and z against the frame number on the right
colours = {"o1": "tab:blue", "o2": "tab:orange", "o3": "tab:green"}
fig = plt.figure(figsize=(16, 7))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
grid = fig.add_gridspec(3, 2)
for name, points in trajectories.items():
    colour = colours[name[:2]]
    ax3d.plot(points[:, 0], points[:, 2], -points[:, 1], color=colour, label=name)   # Same axes as above: x, z (depth), -y (up)
ax3d.set_xlabel("x [m]")
ax3d.set_ylabel("z (depth) [m]")
ax3d.set_zlabel("y (up) [m]")
ax3d.set_title("The three trajectories")
ax3d.legend(fontsize=8)
for axisIndex, axisName in enumerate(("x [m]", "y [m] (down)", "z (depth) [m]")):
    ax = fig.add_subplot(grid[axisIndex, 1])
    for name, points in trajectories.items():
        ax.plot(frames, points[:, axisIndex], color=colours[name[:2]], label=name[:2])
    ax.set_ylabel(axisName)
    if axisIndex == 0:
        ax.legend(ncol=3, fontsize=8)
ax.set_xlabel("frame")
plt.tight_layout()
fig.savefig(outputDir / "all_trajectories.png", dpi=150)

# 3. (c): which intrinsic matrix is more plausible? -----------------------------------------------------
# Z comes directly from the depth sensor, so the matrices only change x and y.
# A pendulum moves on a circular arc in a plane: fit a plane (SVD), then a circle inside it. Small residual = consistent trajectory.
circleFits = {}                                                             # Per matrix: planar coordinates, circle centre, radius
for name, points in trajectories.items():
    centered   = points - points.mean(axis=0)                               # Points relative to their mean
    _, _, vt   = np.linalg.svd(centered, full_matrices=False)               # Rows of vt = principal directions, the first two span the swing plane
    planeAxes  = vt[:2].copy()                                              # The two in-plane directions. The SVD gives them with an arbitrary sign, so we orient them ourselves:
    if planeAxes[0, 0] < 0:
        planeAxes[0] = -planeAxes[0]                                        # First axis points to the right (+x)
    if planeAxes[1, 1] > 0:
        planeAxes[1] = -planeAxes[1]                                        # Second axis points UP (camera y points down, so up = negative y), otherwise the arc can be drawn upside down
    planar     = centered @ planeAxes.T                                     # 2D coordinates inside the best-fit plane
    design     = np.column_stack([2 * planar[:, 0], 2 * planar[:, 1], np.ones(len(planar))])   # Algebraic circle fit: x^2 + y^2 = 2 cx x + 2 cy y + offset
    target     = np.sum(planar ** 2, axis=1)
    (circleX, circleY, offsetTerm), *_ = np.linalg.lstsq(design, target, rcond=None)
    radius     = np.sqrt(offsetTerm + circleX ** 2 + circleY ** 2)          # Fitted pendulum radius
    residual   = np.hypot(planar[:, 0] - circleX, planar[:, 1] - circleY) - radius   # Distance from each point to the circle
    rmsResidual = np.sqrt(np.mean(residual ** 2))
    circleFits[name] = dict(planar=planar, center=np.array([circleX, circleY]), radius=radius)
    print(f"(c) {name}: fitted pendulum radius = {radius:.3f} m, circle-fit RMS residual = {rmsResidual * 1000:.1f} mm")
    if estimatedPendulumLength is not None:
        print(f"    difference to measured string length: {(radius - estimatedPendulumLength) * 1000:.1f} mm")

# 4. (d): learn f_theta(o1) = o2* -----------------------------------------------------------------------
o1Train, o1Test, o2Train, o2Test = train_test_split(o1, o2, test_size=testFraction, random_state=randomSeed)   # Random split into train and test
inputScaler  = StandardScaler().fit(o1Train)                                # Scaling inputs and outputs to mean 0, std 1 helps the network train (scaler is fitted on training data only)
outputScaler = StandardScaler().fit(o2Train)
o1TrainScaled, o2TrainScaled = inputScaler.transform(o1Train), outputScaler.transform(o2Train)
o1TestScaled,  o2TestScaled  = inputScaler.transform(o1Test),  outputScaler.transform(o2Test)

network = MLPRegressor(hidden_layer_sizes=hiddenLayers, 
                       activation="tanh", 
                       solver="adam",  # Trained with Adam (gradient descent)
                       random_state=randomSeed)  
trainLoss, testLoss = [], []                                                # MSE (scaled data) after every epoch, so the learning can be plotted
for epoch in range(maxEpochs):
    network.partial_fit(o1TrainScaled, o2TrainScaled)                       # One epoch: one pass over the training data, the weights theta are updated
    trainLoss.append(mean_squared_error(o2TrainScaled, network.predict(o1TrainScaled)))   # Loss on the data the network trains on
    testLoss.append(mean_squared_error(o2TestScaled, network.predict(o1TestScaled)))      # Loss on held-out data, shows whether it generalises
    if stopLoss is not None and trainLoss[-1] < stopLoss:
        print(f"Early stop after epoch {epoch + 1}: training loss {trainLoss[-1]:.2e} is below {stopLoss:.0e}")
        break
numEpochs = len(trainLoss)                                                  # Epochs actually run

predictO2 = lambda points: outputScaler.inverse_transform(network.predict(inputScaler.transform(points)))   # f_theta in meters: scale in, network, scale back
o2Predicted = predictO2(o1Test)                                             # Prediction on frames the network has not seen

# Loss curves (logarithmic axis, because the loss falls over several orders of magnitude)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(np.arange(1, numEpochs + 1), trainLoss, label="training loss")
ax.semilogy(np.arange(1, numEpochs + 1), testLoss, "--", label="test loss (held out)")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE (scaled data)")
ax.set_title("Training of f_theta")
ax.legend()
fig.savefig(outputDir / "training_loss.png", dpi=150)

print(f"\n(d) Network {hiddenLayers}, {numEpochs} epochs, final loss: training {trainLoss[-1]:.2e}, test {testLoss[-1]:.2e} (MSE, scaled data)")
print("(d) KPIs on the held-out test set:")
for axisIndex, axisName in enumerate("XYZ"):
    rmse = np.sqrt(mean_squared_error(o2Test[:, axisIndex], o2Predicted[:, axisIndex]))
    mae  = mean_absolute_error(o2Test[:, axisIndex], o2Predicted[:, axisIndex])
    r2   = r2_score(o2Test[:, axisIndex], o2Predicted[:, axisIndex])
    print(f"    {axisName}: RMSE={rmse * 1000:.3f} mm, MAE={mae * 1000:.3f} mm, R2={r2:.6f}")

# 5. (e): Euclidean errors ------------------------------------------------------------------------------
o2Star        = predictO2(o1)                                               # Prediction of the network for every valid frame
errorO1O2     = np.linalg.norm(o1 - o2, axis=1)                             # (i)  o1 vs o2
errorO1O2Star = np.linalg.norm(o1 - o2Star, axis=1)                         # (ii) o1 vs o2*
errorModel    = np.linalg.norm(o2 - o2Star, axis=1)                         # Extra: how well o2* reproduces o2
print("\n(e) Euclidean errors:")
for label, errors in (("(i)  |o1 - o2|  ", errorO1O2), ("(ii) |o1 - o2*| ", errorO1O2Star), ("     |o2 - o2*| ", errorModel)):
    mm = errors * 1000                                                      # Meters to millimeters
    print(f"{label}: mean={mm.mean():.1f} mm, median={np.median(mm):.1f} mm, std={mm.std():.1f} mm, max={mm.max():.1f} mm")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(frames, errorO1O2 * 1000, label="|o1 - o2|")
ax.plot(frames, errorO1O2Star * 1000, "--", label="|o1 - o2*|")
ax.plot(frames, errorModel * 1000, label="|o2 - o2*|")
ax.set_xlabel("frame")
ax.set_ylabel("Euclidean error [mm]")
ax.legend()
fig.savefig(outputDir / "euclidean_errors.png", dpi=150)

# 6. (c) visual: what each matrix says compared with the measured pendulum --------------------------------
fig, (axO1, axO2, axO3, axRadius) = plt.subplots(1, 4, figsize=(23, 5))
for ax, (name, fit) in zip((axO1, axO2, axO3), circleFits.items()):
    planar, circleCenter, fittedRadius = fit["planar"], fit["center"], fit["radius"]
    angles    = np.arctan2(planar[:, 1] - circleCenter[1], planar[:, 0] - circleCenter[0])   # Angle of every point seen from the circle centre
    meanAngle = np.angle(np.mean(np.exp(1j * angles)))                      # Circular mean, safe across the +-pi jump
    deviation = np.angle(np.exp(1j * (angles - meanAngle)))                 # Deviation from the mean, wrapped to [-pi, pi]
    arc       = meanAngle + np.linspace(deviation.min() - 0.1, deviation.max() + 0.1, 200)   # Only the part of the circle where the load moved (plus a margin)
    ax.scatter(planar[:, 0], planar[:, 1], c=frames, cmap="viridis", s=8, label="trajectory from this matrix")
    ax.plot(circleCenter[0] + fittedRadius * np.cos(arc), circleCenter[1] + fittedRadius * np.sin(arc),
            color="tab:red", label=f"fitted circle, r = {fittedRadius:.3f} m")
    if estimatedPendulumLength is not None:
        # The real circle: same centre (the pivot is unknown) but the measured string length as radius
        ax.plot(circleCenter[0] + estimatedPendulumLength * np.cos(arc), circleCenter[1] + estimatedPendulumLength * np.sin(arc),
                color="black", linestyle="--", label=f"measured string, L = {estimatedPendulumLength:.3f} m")
    ax.set_aspect("equal")
    ax.set_xlabel("in-plane coordinate 1 [m]")
    ax.set_ylabel("in-plane coordinate 2 [m]")
    ax.set_title(f"{name}: swing plane")
    ax.legend(fontsize=8)

axRadius.bar([name[:2] for name in circleFits.keys()], [fit["radius"] for fit in circleFits.values()], color=["tab:blue", "tab:orange", "tab:green"])   # Radius implied by each matrix
if estimatedPendulumLength is not None:
    axRadius.axhline(estimatedPendulumLength, color="black", linestyle="--", label=f"measured string length {estimatedPendulumLength:.3f} m")
    axRadius.legend()
axRadius.set_ylabel("pendulum radius [m]")
axRadius.set_title("Radius implied by each matrix")
plt.tight_layout()
fig.savefig(outputDir / "swing_plane_vs_measured.png", dpi=150)

plt.show()