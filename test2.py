import pyrender

focal = [616.640869140625, 616.2581787109375]
princpt = [308.548095703125, 248.52310180664062]

camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
print(camera.get_projection_matrix(640, 480))
print(5.1885790117450188e02)