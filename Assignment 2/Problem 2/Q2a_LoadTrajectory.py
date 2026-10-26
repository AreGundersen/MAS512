from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from ouster.sdk import open_source, client

# Directories and paths
HERE           = Path(__file__).resolve().parent
recordingDir   = HERE / "lidar-pcd processing-2026 class"    # Folder with the recordings; every .pcap needs a .json with the same base name next to it
outputDir      = HERE / "Q2_output"                          # Plots, CSV and NPZ files are saved here
swingingPcap   = recordingDir / "swinging_load.pcap"         # Recording of the swinging load (Q2 a)
stationaryPcap = recordingDir / "stationary_load.pcap"       # Recording of the load hanging still, only used to estimate the noise of the centre (None = skip)

# Parameters
minValidPoints = 20000                                       # Frames with fewer valid points are incomplete (the sensor starts in the middle of a rotation) and are skipped
roiMin         = np.array([0.0, 0.8, -0.3])                  # Region of interest in the sensor frame (x forward, y left, z up) in meters, lower corner
roiMax         = np.array([1.6, 3.0, 1.2])                   # Upper corner. The box contains the whole swing of the load but not the walls, floor and equipment of the lab
dbscanEps      = 0.08                                        # DBSCAN: points closer than this (m) belong to the same cluster
dbscanMinPts   = 10                                          # DBSCAN: minimum number of points in a cluster
bodyHeight     = 0.26                                        # The load is a block of about 0.3 m; the sling and hook above it are cut away (they come and go and make the centre z jump)
gravity        = 9.81                                        # m/s^2, used for the pendulum length
saveHtml       = True                                        # Also save an interactive 3D plot (plotly) that can be rotated

outputDir.mkdir(parents=True, exist_ok=True)                                # Creating the output folder if it does not exist

recordings = {"swinging": swingingPcap}                                     # Recordings to process, name -> path
if stationaryPcap is not None:
    recordings["stationary"] = stationaryPcap

results = {}                                                                # Per recording: frame numbers, times, centres, number of points and the points themselves
firstFramePoints = None                                                     # One full frame, only used for the region-of-interest picture

# 1. Find the load in every frame ---------------------------------------------------------------------
for name, pcapPath in recordings.items():
    source = open_source(str(pcapPath))                                     # Opening the pcap (the .json with the same name is read automatically)
    sensorInfo = source.metadata[0] if isinstance(source.metadata, (list, tuple)) else source.metadata   # Sensor metadata (beam angles etc.)
    xyzLookup = client.XYZLut(sensorInfo)                                   # Lookup table that turns the range image into x, y, z in meters

    frameNumbers, timesS, centers, pointsPerFrame, clouds = [], [], [], [], []
    for frameIdx, scan in enumerate(source):
        allPoints = xyzLookup(scan).reshape(-1, 3)                          # (64 x 1024, 3) points in the sensor frame, (0, 0, 0) means no return
        if np.count_nonzero(np.any(allPoints != 0, axis=1)) < minValidPoints:
            continue                                                        # Incomplete frame
        if name == "swinging" and firstFramePoints is None:
            firstFramePoints = allPoints[np.any(allPoints != 0, axis=1)]    # Saving the first complete frame for the ROI picture

        insideBox = np.all((allPoints >= roiMin) & (allPoints <= roiMax), axis=1)   # True for points inside the region of interest
        boxPoints = allPoints[insideBox]                                    # Only these points are used, the rest of the lab is ignored

        cloud  = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boxPoints))     # open3d point cloud, needed for DBSCAN
        labels = np.array(cloud.cluster_dbscan(eps=dbscanEps, min_points=dbscanMinPts))   # Cluster number per point, -1 = noise
        if len(labels) == 0 or labels.max() < 0:
            print(f"Frame {frameIdx}: no cluster inside the region of interest")
            continue
        sizes      = np.bincount(labels[labels >= 0])                       # Number of points in each cluster
        loadPoints = boxPoints[labels == sizes.argmax()]                    # The biggest cluster is the load

        bottomZ    = np.percentile(loadPoints[:, 2], 2)                     # Lowest part of the load (2nd percentile so one stray point does not decide)
        loadPoints = loadPoints[loadPoints[:, 2] <= bottomZ + bodyHeight]   # Cutting away the sling and hook above the body

        if clouds and np.array_equal(loadPoints, clouds[-1]):
            print(f"Frame {frameIdx}: identical to the previous frame, skipped")   # The last frame of a recording can be a repeat of the one before
            continue

        stamps = np.asarray(scan.timestamp)                                 # One timestamp per column (nanoseconds)
        frameNumbers.append(frameIdx)
        timesS.append(float(np.median(stamps[stamps > 0])) * 1e-9)          # Time of the frame = median column time, in seconds
        centers.append(loadPoints.mean(axis=0))                             # Centre of the load = mean of the points (centroid)
        pointsPerFrame.append(len(loadPoints))
        clouds.append(loadPoints)

    timesS = np.array(timesS) - timesS[0]                                   # Time axis starting at zero
    results[name] = dict(frames=np.array(frameNumbers), time=timesS, centers=np.array(centers),
                         counts=np.array(pointsPerFrame), clouds=clouds)
    print(f"{name}: {len(centers)} usable frames, {min(pointsPerFrame)} to {max(pointsPerFrame)} points in the cluster")

swing        = results["swinging"]
frameNumbers = swing["frames"]
timesS       = swing["time"]
centers      = swing["centers"]                                             # (number of frames, 3) centre of the load in the sensor frame

# 2. Save the result ----------------------------------------------------------------------------------
np.savetxt(outputDir / "load_centroids.csv", np.column_stack([frameNumbers, timesS, centers, swing["counts"]]),
           delimiter=",", header="frame_index,time_s,x_m,y_m,z_m,cluster_points", comments="",
           fmt=["%d", "%.4f", "%.5f", "%.5f", "%.5f", "%d"])                # Centroids, read by Q2b for comparison
np.savez_compressed(outputDir / "load_points.npz", frame_indices=frameNumbers, time_s=timesS,
                    points_per_frame=swing["counts"], all_points=np.concatenate(swing["clouds"]))   # All load points, Q2b runs KISS-ICP on these

# 3. Picture of the region of interest ----------------------------------------------------------------
fig, (axTop, axSide) = plt.subplots(1, 2, figsize=(13, 6))
nearPoints = firstFramePoints[np.linalg.norm(firstFramePoints, axis=1) < 8.0]   # Only points closer than 8 m, to keep the picture readable
for ax, a, b, labelA, labelB in ((axTop, 0, 1, "x", "y"), (axSide, 0, 2, "x", "z")):
    ax.scatter(nearPoints[:, a], nearPoints[:, b], s=0.5, c="lightgrey")
    ax.plot([roiMin[a], roiMax[a], roiMax[a], roiMin[a], roiMin[a]],
            [roiMin[b], roiMin[b], roiMax[b], roiMax[b], roiMin[b]], color="red", label="region of interest")   # Box drawn as a red rectangle
    ax.set_aspect("equal")
    ax.set_xlabel(f"{labelA} [m]")
    ax.set_ylabel(f"{labelB} [m]")
    ax.legend()
axTop.set_title("Top view of one frame (points closer than 8 m)")
axSide.set_title("Side view")
plt.tight_layout()
fig.savefig(outputDir / "roi_check.png", dpi=150)

# 4. Trajectory plots ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")                            # Sensor frame: x forward, y left, z up
ax3d.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="lightgrey", linewidth=0.8)
sc = ax3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=timesS, cmap="viridis", s=18)   # Colour = time
ax3d.set_xlabel("x [m] (forward)")
ax3d.set_ylabel("y [m] (left)")
ax3d.set_zlabel("z [m] (up)")
ax3d.set_title("Centre of the swinging load, Ouster sensor frame")
fig.colorbar(sc, ax=ax3d, label="time [s]", shrink=0.6)
axTime = fig.add_subplot(1, 2, 2)
for col, axisName in enumerate("xyz"):
    axTime.plot(timesS, centers[:, col], label=axisName)                    # One line per axis
axTime.set_xlabel("time [s]")
axTime.set_ylabel("position [m]")
axTime.set_title("Centre of the load versus time")
axTime.legend()
plt.tight_layout()
fig.savefig(outputDir / "load_trajectory.png", dpi=150)

# Same trajectory in the axes of the assignment: x to the right, y up, z toward the viewer
# (viewer behind the sensor looking forward): x_right = -sensor y, y_up = sensor z, z_toward_viewer = -sensor x
xRight        = -centers[:, 1]
yUp           = centers[:, 2]
zTowardViewer = -centers[:, 0]
fig = plt.figure(figsize=(11, 7))
axA = fig.add_subplot(1, 1, 1, projection="3d")                             # matplotlib draws its third axis vertically, so y (up) is given last
sc = axA.scatter(xRight, zTowardViewer, yUp, c=timesS, cmap="viridis", s=18)
axA.plot(xRight, zTowardViewer, yUp, color="lightgrey", linewidth=0.8)
axA.set_xlabel("x [m] (right)", labelpad=12)
axA.set_ylabel("z [m] (toward viewer)", labelpad=12)
axA.set_zlabel("y [m] (up)", labelpad=8)
axA.set_box_aspect((3, 1, 1))                                               # Not true scale: the swing is much longer than it is deep or high
axA.view_init(elev=20, azim=-70)
axA.locator_params(nbins=4)
axA.set_title("3D trajectory of the load centre (x right, y up, z toward the viewer)")
fig.colorbar(sc, ax=axA, label="time [s]", shrink=0.6, pad=0.15)
fig.savefig(outputDir / "load_trajectory_assignment_axes.png", dpi=150)

if saveHtml:
    figHtml = go.Figure(data=[go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="lines+markers",
                                           marker=dict(size=3, color=timesS, colorscale="Viridis", colorbar=dict(title="time [s]")),
                                           line=dict(color="lightgrey", width=2))])
    figHtml.update_layout(scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]", aspectmode="data"),
                          title="Centre of the swinging load")
    figHtml.write_html(str(outputDir / "load_trajectory.html"))             # Interactive version, open in a browser

# 5. Numbers for the report ---------------------------------------------------------------------------
print("\nCentre of the load, range over the recording:")
for col, axisName in enumerate("xyz"):
    print(f"  {axisName}: {centers[:, col].min():.3f} to {centers[:, col].max():.3f} m (peak to peak {1000 * np.ptp(centers[:, col]):.0f} mm)")

centered    = centers - centers.mean(axis=0)                                # Centres relative to their mean
_, _, axes  = np.linalg.svd(centered, full_matrices=False)                  # First principal axis = direction with the largest spread = swing direction
swingDir    = axes[0]
alongSwing  = centered @ swingDir                                           # Position along the swing direction, a sinusoid for a pendulum

sinusoid    = lambda t, offset, amplitude, period, phase: offset + amplitude * np.sin(2 * np.pi * t / period + phase)   # Model for the fit
spectrum    = np.abs(np.fft.rfft(alongSwing - alongSwing.mean(), n=8 * len(alongSwing)))   # Spectrum, only used for the starting guess of the period
freqs       = np.fft.rfftfreq(8 * len(alongSwing), d=np.median(np.diff(timesS)))
freqGuess   = freqs[1:][spectrum[1:].argmax()]                              # Strongest frequency (skipping the zero frequency)
params, _   = curve_fit(sinusoid, timesS, alongSwing, p0=[0, 0.5 * np.ptp(alongSwing), 1.0 / freqGuess, 0.0])
period      = abs(params[2])                                                # Period in seconds
amplitude   = abs(params[1])                                                # Amplitude in meters
fitRms      = np.sqrt(np.mean((alongSwing - sinusoid(timesS, *params)) ** 2))   # How well a sinusoid describes the motion
pendulumLen = gravity * period ** 2 / (4 * np.pi ** 2)                      # Small-angle pendulum: T = 2 pi sqrt(L / g)  ->  L = g T^2 / (4 pi^2)
print(f"\nSwing direction (unit vector in the sensor frame): {np.round(swingDir, 3)}")
print(f"Sinusoid fit: period = {period:.2f} s, amplitude = {amplitude * 1000:.0f} mm, residual (RMS) = {fitRms * 1000:.1f} mm")
print(f"Pendulum length from the period: {pendulumLen:.2f} m")

if "stationary" in results:
    still = results["stationary"]["centers"]                                # Centres with the load hanging still
    stdMm = 1000 * still.std(axis=0)                                        # Spread = noise of the centre estimate
    print(f"\nStationary load: {len(still)} frames, centre {np.round(still.mean(axis=0), 3)} m, std x/y/z = {stdMm[0]:.1f} / {stdMm[1]:.1f} / {stdMm[2]:.1f} mm")

plt.show()
