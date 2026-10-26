import argparse
import csv
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from ultralytics import YOLO


# KITTI I/O
def read_calib(path):
    d = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = d["P2"].reshape(3, 4)                      # camera 2 (left color), M_int
    R0 = np.eye(4)
    R0[:3, :3] = d["R0_rect"].reshape(3, 3)         # rectification
    Tr = np.eye(4)
    Tr[:3, :4] = d["Tr_velo_to_cam"].reshape(3, 4)  # lidar -> camera 0, M_ext
    return P2, R0, Tr


def read_velodyne(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # x, y, z, reflectance


def read_labels(path):
    objs = []
    if not os.path.exists(path):
        return objs
    with open(path) as f:
        for line in f:
            p = line.split()
            if p[0] == "DontCare":
                continue
            x1, y1, x2, y2 = map(float, p[4:8])
            X, Y, Z = map(float, p[11:14])          # camera coordinates (bottom centre of the 3D box)
            objs.append(dict(cls=p[0], box=(x1, y1, x2, y2), X=X, Y=Y, Z=Z,
                             rng=float(np.sqrt(X**2 + Y**2 + Z**2))))
    return objs


#projection
def to_camera(pts_xyz, R0, Tr):
    xyz1 = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1))]).T      
    return (R0 @ Tr @ xyz1)[:3].T                                


def to_pixels(cam_xyz, P2):
    cam1 = np.hstack([cam_xyz, np.ones((len(cam_xyz), 1))]).T
    uvw = P2 @ cam1
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


def in_image(cam_xyz, u, v, shape):
    h, w = shape[:2]
    return (cam_xyz[:, 2] > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)


# lidar processing (Open3D)
def make_pcd(xyz):
    pcd = o3d.geometry.PointCloud()                 # point cloud object
    pcd.points = o3d.utility.Vector3dVector(xyz)    # assign points
    return pcd


def cluster_centroid(pts, eps=0.5, min_points=5):
    if len(pts) < min_points:
        return None, None
    labels = np.array(make_pcd(pts).cluster_dbscan(eps=eps, min_points=min_points))  # -1 = noise
    if labels.size == 0 or labels.max() < 0:
        return None, None
    sizes = np.bincount(labels[labels >= 0])
    best = int(np.argmax(sizes))
    cl = pts[labels == best]
    return cl, np.mean(cl, axis=0)


# helpers
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# yolo and kitti class' names are not the same, therefore:
GROUP = {"car": "veh", "truck": "veh", "bus": "veh", "Car": "veh", "Van": "veh",
         "Truck": "veh", "Tram": "veh",
         "person": "ped", "Pedestrian": "ped", "Person_sitting": "ped",
         "bicycle": "cyc", "motorcycle": "cyc", "Cyclist": "cyc"}


# find centroid
def box_distance(det_box, pts_xyz, u, v, shrink=0.5):
    x1, y1, x2, y2 = det_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    m = (u > cx - hw) & (u < cx + hw) & (v > cy - hh) & (v < cy + hh)
    cl, cen = cluster_centroid(pts_xyz[m])
    if cl is None:
        return None, None, int(m.sum()), None, None
    return float(cen[2]), float(np.linalg.norm(cen)), len(cl), cen, cl


#plotting
def plot_projection(img_rgb, u, v, z, path):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.imshow(img_rgb)
    sc = ax.scatter(u, v, c=z, s=3, cmap="jet", vmin=0, vmax=60)
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, label="Distance z [m]")
    ax.set_title("Lidar projected onto the camera image")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bev(pts, img_rgb, u, v, idx, path):
    cols = img_rgb[v.astype(int), u.astype(int)] / 255.0
    p = pts[idx]
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(-p[:, 1], p[:, 0], c=cols, s=2)      # lidar: x forward, y left -> plot -y to the right
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.set_xlabel("left  <->  right [m]")
    ax.set_ylabel("forward [m]")
    ax.set_title("Lidar points coloured with camera image (BEV)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_clusters(cam_np, dets, path):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(cam_np[:, 0], cam_np[:, 2], s=1, c="lightgrey", label="lidar points")
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(dets):
        if d["cluster"] is None:
            continue
        c = d["cluster"]
        ax.scatter(c[:, 0], c[:, 2], s=6, color=cmap(i % 10))
        ax.scatter(d["centroid"][0], d["centroid"][2], marker="x", s=80, color="k")
        ax.text(d["centroid"][0] + 0.5, d["centroid"][2], f'{d["name"]} {d["depth"]:.1f} m', fontsize=8)
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 60)
    ax.set_aspect("equal")
    ax.set_xlabel("camera x: left <-> right [m]")
    ax.set_ylabel("camera z: forward [m]")
    ax.set_title("DBSCAN clusters with centroids")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detections(img_bgr, dets, path):
    img = img_bgr.copy()
    cmap = plt.get_cmap("jet")
    for d in dets:
        if d["depth"] is None:
            continue
        col = tuple(int(255 * c) for c in cmap(min(d["depth"] / 60.0, 1.0))[:3][::-1])  # RGB->BGR
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f'{d["name"]} {d["depth"]:.1f} m'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 60))
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, label="Distance [m]")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\ssgje\Documents\School\MAS512 - Computer Vision and Embedded AI\datasets\archive") # THIS IS ONLY FOR SEBASTIAN, CHANGE TO YOUR PATH
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel_down_sample size [m]")
    ap.add_argument("--ids", nargs="*", help="use these image ids instead of a random sample")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = os.path.join(args.root, "training")
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(tr, "image_2")) if f.endswith(".png"))
    random.seed(args.seed)
    chosen = args.ids if args.ids else random.sample(ids, args.n)
    print("Chosen images:", chosen)

    model = YOLO(args.model)
    rows = []

    for sid in chosen:
        img_bgr = cv2.imread(os.path.join(tr, "image_2", sid + ".png"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pts = read_velodyne(os.path.join(tr, "velodyne", sid + ".bin"))
        P2, R0, Tr = read_calib(os.path.join(tr, "calib", sid + ".txt"))
        gts = read_labels(os.path.join(tr, "label_2", sid + ".txt"))

        # (i) projection 
        cam_all = to_camera(pts[:, :3], R0, Tr)
        u_all, v_all = to_pixels(cam_all, P2)
        ok = in_image(cam_all, u_all, v_all, img_bgr.shape)
        idx = np.where(ok)[0]
        plot_projection(img_rgb, u_all[ok], v_all[ok], cam_all[ok, 2],
                        os.path.join(args.out, f"fusion_{sid}_projection.png"))
        plot_bev(pts, img_rgb, u_all[ok], v_all[ok], idx, os.path.join(args.out, f"fusion_{sid}_bev.png"))

        # lidar processing 
        pcd = make_pcd(cam_all[ok])                                 
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)          
        cam_np = np.asarray(pcd.points)                           
        u, v = to_pixels(cam_np, P2)
        print(sid, f"{len(cam_np)} points after downsampling")

        # (ii) detection + distance
        res = model(img_bgr, conf=args.conf, verbose=False)[0]
        dets = []
        for b in res.boxes:
            name = res.names[int(b.cls)]
            if name not in GROUP:
                continue
            box = tuple(float(x) for x in b.xyxy[0])
            depth, rng, npts, cen, cl = box_distance(box, cam_np, u, v)
            dets.append(dict(name=name, box=box, conf=float(b.conf), depth=depth, rng=rng,
                             npts=npts, centroid=cen, cluster=cl))
        plot_detections(img_bgr, dets, os.path.join(args.out, f"fusion_{sid}_detection.png"))
        plot_clusters(cam_np, dets, os.path.join(args.out, f"fusion_{sid}_clusters.png"))

        # (iii) validation 
        for d in dets:
            if d["depth"] is None:
                continue
            best, best_iou = None, 0.3
            for g in gts:
                if GROUP.get(g["cls"]) != GROUP[d["name"]]:
                    continue
                o = iou(d["box"], g["box"])
                if o > best_iou:
                    best, best_iou = g, o
            if best is None:
                continue
            rows.append(dict(image=sid, cls=d["name"], iou=round(best_iou, 2), n_lidar_pts=d["npts"],
                             est_depth=round(d["depth"], 2), gt_depth=round(best["Z"], 2),
                             err_depth=round(d["depth"] - best["Z"], 2),
                             est_range=round(d["rng"], 2), gt_range=round(best["rng"], 2),
                             err_range=round(d["rng"] - best["rng"], 2)))

    if rows:
        with open(os.path.join(args.out, "distance_validation.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        e = np.array([r["err_depth"] for r in rows])
        print(f"\n{len(rows)} objects validated")
        print(f"Depth error: MAE={np.abs(e).mean():.2f} m, RMSE={np.sqrt((e**2).mean()):.2f} m, "
              f"bias={e.mean():+.2f} m")
        for r in rows:
            print(r)
    else:
        print("No detections matched the ground truth. Try another --seed or a lower --conf.")


if __name__ == "__main__":
    main()
