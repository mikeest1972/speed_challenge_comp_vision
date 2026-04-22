import cv2
import numpy as np
from tqdm import tqdm
# good ref https://www.geeksforgeeks.org/python/opencv-the-gunnar-farneback-optical-flow/

# read the video
WIDTH = 320
HEIGHT = 240
capture = cv2.VideoCapture('../data/train.mp4')
_, frame1 = capture.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prvs = cv2.resize(prvs, (WIDTH, HEIGHT))
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
h, w = prvs.shape
# prealocate output on disk so flows are written directly withotu accumultin in ram
flows = np.lib.format.open_memmap('../data/precomputed_flows.npy', mode='w+', dtype=np.float32, shape=(total_frames - 1, h, w, 2))
for i in tqdm(range(total_frames - 1)):
    ret, frame2 = capture.read()
    if not ret:
        break
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.resize(next_frame, (WIDTH, HEIGHT))
    # save precomputed flows to file to speed up traning
    flows[i] = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prvs = next_frame
    





    # visualization not needed to precompute the flow, but it is useful to check if the flow is being computed correctly
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite(f'/app/data/flow_preview_{frame_count}.png', rgb)
    # frame_count += 1
    # if frame_count > 5:
    #     break
    # prvs = next


