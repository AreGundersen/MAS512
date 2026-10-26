from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

# Directories and paths (run Q2a_LoadTrajectory.py first, it writes the files used here)
HERE      = Path(__file__).resolve().parent
outputDir = HERE / "Q2_output"
pointsFile = outputDir / "load_points.npz"                  # Load points per frame, from Q2a

# Parameters
# The defaults of KISS-ICP are made for a car in a street (ranges 5-100 m, voxels about 1 m).
# Here the scene is one small object of about 0.3 m, so everything is scaled down.
minRange        = 0.0       # Points closer than this to the origin are removed (the load is centred on the origin below)
maxRange        = 10.0      # Points further away than this are removed
voxelSize       = 0.02      # Voxel size (m) of the local map and of the downsampling of each frame
initialTresh    = 0.2       # First guess (m) of how far a point moves between two frames, the adaptive threshold starts here
minMotionTresh  = 0.01      # The adaptive threshold never goes below this (m)
maxPtsPerVoxel  = 20        # Maximum number of points stored in one voxel of the map
deskew          = False     # No motion compensation inside one rotation, the load is slow compared to 10 fps
axisLength      = 0.08      # Length (m) of the coordinate axes drawn in the pose plot

# 1. Load the points from Q2a -------------------------------------------------------------------------
data       = np.load(pointsFile)
allPoints  = data["all_points"]                                             # All load points of all frames after each other
counts     = data["points_per_frame"]                                       # Number of points in each frame
timesS     = data["time_s"]                                                 # Time of each frame in seconds
bounds     = np.concatenate([[0], np.cumsum(counts)])                       # Start and end index of every frame in allPoints
frames     = [allPoints[bounds[i]:bounds[i + 1]] for i in range(len(counts))]   # List with the points of each frame
centroids  = np.array([frame.mean(axis=0) for frame in frames])             # Centroid per frame (the Q2a result), used for comparison
numFrames  = len(frames)

firstCenter = centroids[0].copy()                                           # Centre of the load in frame 0
frames      = [frame - firstCenter for frame in frames]                     # Moving the load to the origin, where KISS-ICP expects the sensor to start

# 2. Run KISS-ICP -------------------------------------------------------------------------------------
config = KISSConfig()                                                       # Default configuration, changed below
config.data.min_range                    = minRange
config.data.max_range                    = maxRange
config.data.deskew                       = deskew
config.mapping.voxel_size                = voxelSize
config.mapping.max_points_per_voxel      = maxPtsPerVoxel
config.adaptive_threshold.initial_threshold = initialTresh
config.adaptive_threshold.min_motion_th  = minMotionTresh
odometry = KissICP(config)                                                  # The odometry object keeps the local map between frames

poses = []                                                                  # One 4x4 pose per frame
for frame in frames:
    odometry.register_frame(frame.astype(np.float64), np.empty(0))          # Aligns the frame to the map and adds it (no timestamps needed, deskew is off)
    poses.append(odometry.last_pose.copy())                                 # Estimated pose of the "sensor" in frame k relative to frame 0
poses = np.array(poses)                                                     # (numFrames, 4, 4)

# 3. From KISS-ICP pose to the pose of the load -------------------------------------------------------
# KISS-ICP assumes that the SENSOR moves through a fixed world. Here the sensor is fixed and the LOAD moves.
# "Sensor moved by T" is the same as "scene moved by T^-1", so the pose of the load relative to frame 0 is inverse(T).
loadPoses     = np.array([np.linalg.inv(pose) for pose in poses])           # Pose of the load in every frame
loadPositions = loadPoses[:, :3, 3] + firstCenter                           # Position of the load centre (the origin of the load maps to the translation part)
traces        = np.array([np.trace(pose[:3, :3]) for pose in loadPoses])    # Trace of the rotation matrix = 1 + 2 cos(angle)
rotationDeg   = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))     # Rotation of the load relative to frame 0 in degrees

# 4. Compare with the centroids from Q2a --------------------------------------------------------------
diffMm = 1000 * np.linalg.norm(loadPositions - centroids, axis=1)           # 3D distance between KISS-ICP and centroid in mm
print(f"Frames processed: {numFrames}")
print(f"Difference KISS-ICP vs centroid (3D): mean {diffMm.mean():.1f} mm, max {diffMm.max():.1f} mm, RMS {np.sqrt(np.mean(diffMm ** 2)):.1f} mm")
print(f"Largest rotation relative to frame 0: {rotationDeg.max():.1f} degrees")
for k in range(3):
    print(f"\nPose of the load in frame {k}:\n{np.round(loadPoses[k], 4)}")  # The first poses as 4x4 matrices

np.savez(outputDir / "kiss_icp_poses.npz", poses_sensor_motion=poses, load_poses=loadPoses, time_s=timesS)
np.savetxt(outputDir / "kiss_icp_positions.csv", np.column_stack([timesS, loadPositions, rotationDeg]),
           delimiter=",", header="time_s,x_m,y_m,z_m,rotation_deg", comments="", fmt="%.5f")

# 5. Plots --------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")                            # 3D trajectory, KISS-ICP against centroid
ax3d.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], color="grey", linewidth=1.5, label="centroid (Q2a)")
ax3d.plot(loadPositions[:, 0], loadPositions[:, 1], loadPositions[:, 2], color="tab:red", linestyle="--", linewidth=1.5, label="KISS-ICP")
ax3d.scatter(*centroids[0], color="black", s=40, label="start")
ax3d.set_xlabel("x [m] (forward)")
ax3d.set_ylabel("y [m] (left)")
ax3d.set_zlabel("z [m] (up)")
ax3d.set_title("Position of the load, KISS-ICP against centroid")
ax3d.legend()
grid = fig.add_gridspec(3, 2)
for col, axisName in enumerate("xyz"):                                      # One panel per axis on the right side
    ax = fig.add_subplot(grid[col, 1])
    ax.plot(timesS, centroids[:, col], color="grey", label="centroid (Q2a)")
    ax.plot(timesS, loadPositions[:, col], color="tab:red", linestyle="--", label="KISS-ICP")
    ax.set_ylabel(f"{axisName} [m]")
    if col == 0:
        ax.legend(fontsize=8)
        ax.set_title("Position of the load versus time")
ax.set_xlabel("time [s]")
plt.tight_layout()
fig.savefig(outputDir / "kiss_icp_trajectory.png", dpi=150)

fig = plt.figure(figsize=(7, 7))                                            # Pose plot: the rotation part of the pose as three axes
axP = fig.add_subplot(1, 1, 1, projection="3d")
axP.plot(loadPositions[:, 0], loadPositions[:, 1], loadPositions[:, 2], color="tab:red", linestyle="--", linewidth=1)
for k in range(0, numFrames, 10):                                           # Every 10th frame
    origin   = loadPositions[k]
    rotation = loadPoses[k][:3, :3]                                         # Columns = x, y, z axes of the load
    for col, colour in enumerate(("r", "g", "b")):
        tip = origin + rotation[:, col] * axisLength
        axP.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], color=colour, linewidth=1.5)
axP.set_xlabel("x [m] (forward)")
axP.set_ylabel("y [m] (left)")
axP.set_zlabel("z [m] (up)")
axP.set_title("Pose of the load (x red, y green, z blue axes of the load, every 10th frame)")
fig.savefig(outputDir / "kiss_icp_poses.png", dpi=150)

fig, (axDiff, axRot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)      # Difference and rotation versus time
axDiff.plot(timesS, diffMm)
axDiff.set_ylabel("|KISS-ICP - centroid| [mm]")
axDiff.set_title("Difference between KISS-ICP and the centroid")
axRot.plot(timesS, rotationDeg)
axRot.set_ylabel("rotation of the load [deg]")
axRot.set_xlabel("time [s]")
axRot.set_title("Rotation of the load relative to frame 0 (KISS-ICP)")
plt.tight_layout()
fig.savefig(outputDir / "kiss_icp_difference.png", dpi=150)
plt.show()
