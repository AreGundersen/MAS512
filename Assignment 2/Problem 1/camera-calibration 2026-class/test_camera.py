import pyrealsense2 as rs

# Resolution of the stream; use the same one as the swinging-load video
WIDTH = 640
HEIGHT = 480
FPS = 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# The factory calibration of the colour camera in exactly this stream mode
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pipeline.stop()

print(f"fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.ppx}, cy={intrinsics.ppy}")
print("Distortion model:", intrinsics.model, "| coefficients:", intrinsics.coeffs)