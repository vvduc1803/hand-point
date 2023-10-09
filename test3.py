import pyrealsense2 as rs
pipe = rs.pipeline()
profile = pipe.start()
try:
  for i in range(0, 100):
    frames = pipe.wait_for_frames()
    for f in frames:
      print(f.profile)
finally:
    pipe.stop()
dpt_frame = pipe.wait_for_frames().get_depth_frame().as_depth_frame()
pixel_distance_in_meters = dpt_frame.get_distance(x,y)