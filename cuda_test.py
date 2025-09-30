import cv2
print(cv2.__version__)                        # should say 4.10.0
print(cv2.cuda.getCudaEnabledDeviceCount())   # should be >= 1
print(cv2.getBuildInformation())

import numpy as np, cv2
img = (np.random.rand(1080,1920)*255).astype(np.uint8)
gpu = cv2.cuda_GpuMat(); gpu.upload(img) # type: ignore
canny = cv2.cuda.createCannyEdgeDetector(50,150)
edges = canny.detect(gpu); _ = edges.download()
print("GPU Canny OK")