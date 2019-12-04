import argparse
# import pyflow
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import scipy

parser = argparse.ArgumentParser(
  description="Optical flow debugger by warpping the images")
parser.add_argument(
  '--image_dir', dest='image_dir', help='Input the image directories', required=True)
parser.add_argument(
  '--forward_flow', dest='forward_flow', help='Input the path to forward flow .npy format file', required=True)
parser.add_argument(
  '--backward_flow', dest='backward_flow', help='Input the path to backward flow .npy format file', required=True)
args = parser.parse_args()


def bilinear_interpolation(flow, x_pos, y_pos):
  '''Interpolate (x, y) from values associated with four points.
  Four points are in list of four triplets : (x_pos, y_pos, value)
  - x_pos and y_pos is the pixels position on the images
  - flow is the matrix at x_pos and y_pos to be a reference for interpolated point.
  Ex : x_pos = 12, y_pos = 5.5
       # flow is the matrix of source to be interpolated
       # Need to form in 4 points that a rectangle shape
       flow_location = [(10, 4, 100), === [(x1, y1, value_x1y1),
                        (20, 4, 200), ===  (x2, y1, value_x2y1),
                        (10, 6, 150), ===  (x1, y2, value_x1y2),
                        (20, 6, 300)] ===  (x2, y2, value_x2y2)]
      Reference : https://en.wikipedia.org/wiki/Bilinear_interpolation

  '''
  # Create a flow_location : clip the value from 0-flow.shape[0-or-1]-1
  x1 = np.clip(int((np.floor(x_pos))), 0, flow.shape[0]-1)
  x2 = np.clip(x1 + 1, 0, flow.shape[0]-1)
  y1 = np.clip(int((np.floor(y_pos))), 0, flow.shape[1]-1)
  y2 = np.clip(y1 + 1, 0, flow.shape[1]-1)
  x_pos = np.clip(x_pos, 0, flow.shape[0]-1)
  y_pos = np.clip(y_pos, 0, flow.shape[1]-1)

  # Last pixels will be the problem
  if x1 == flow.shape[0]-1:
    x1 = flow.shape[0]-2
  if y1 == flow.shape[0]-1:
    y1 = flow.shape[0]-2

  print("X : ", x_pos, x1, x2)
  print("Y : ", y_pos, y1, y2)
  # print(flow[x1, x2])
  flow_area = [(x1, y1, flow[x1][y1]),
               (x2, y1, flow[x2][y1]),
               (x1, y2, flow[x1][y2]),
               (x2, y2, flow[x2][y2])]

  print("Flow interesting area : ", flow_area)
  flow_area = sorted(flow_area)
  (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = flow_area

  if x1!=_x1 or x2!= _x2 or y1!=_y1 or y2!=_y2:
    raise ValueError('Given grid do not form a rectangle.')
  if not x1 <= x_pos <= x2 or not y1 <= y_pos <= y2:
    raise ValueError('(x, y) that want to interpolated is not within the rectangle')

  return ((q11 * (x2-x_pos) * (y2-y_pos)) +
          (q21 * (x_pos-x1) * (y2-y_pos)) +
          (q12 * (x2-x_pos) * (y_pos-y1)) +
          (q22 * (x_pos-x1) * (y_pos-y1))
          / ((x2-x1) * (y2-y1)))

def get_warping_img(img1, img2, forward_flow, backward_flow):
  forward_warp_img = forward_warp()
  backward_warp_img = backward_warp()
  return forward_warp_img, backward_warp_img

def compose_flow(v0, v1):
  ''' This function take 2 flows and compose it
      v0 : ref <- a1
      v1 : a1 <- a2
      Full path : ref <- a1 <- a2
      output : ref <- a2
  '''
  # For compose any 2 adjacent flow together.
  composed_flow = np.zeros((v0.shape[0], v0.shape[1], 2))
  print(composed_flow.shape)
  for i in range(v0.shape[0]):
    for j in range(v0.shape[1]):
      # Find the landed pixels locations from v1
      nr = i + v0[i][j][1]  # On y-axis
      nc = j + v0[i][j][0]  # On x-axis
      # Flow on x-axis
      composed_flow[i][j][0] = v0[i][j][0] + bilinear_interpolation(v1[..., 0], nr, nc)
      # Flow on y-axis
      composed_flow[i][j][1] = v0[i][j][1] + bilinear_interpolation(v1[..., 1], nr, nc)
  print("Composed flow : ", composed_flow)
  print("V0 : ", v0)
  print("V1 : ", v1)
  return composed_flow

if __name__ == "__main__":
  image_dir = args.image_dir
  forward_flow = np.load(args.forward_flow)
  backward_flow = np.load(args.backward_flow)
  # For re-check this is not the same flow from images
  print(forward_flow - backward_flow)
  print("Forward flow ({}) : ".format(forward_flow.shape, forward_flow))
  print("Backward flow ({}) : ".format(backward_flow.shape, backward_flow))
  image_filelist = sorted(glob.glob(image_dir + '/frame*.png'))
  while True:
    # This loop iterate until finishins all of pixles.
    print("Warping forward/backward between 2 given images")
    i = int(input("Frame i-th of input images : "))
    j = int(input("Frame j-th of input images : "))
    if (i<0) or (j<0):
      exit()

    # Read a img1 and img2
    img1 = cv2.imread(image_filelist[i])[..., ::-1]
    img2 = cv2.imread(image_filelist[j])[..., ::-1]

    # Warping image : Forward warp(img1->img2) and Backward warp(img2->img1)
    # forward_warp_img, backward_warp_img = get_warping_img()
    compose_flow(forward_flow[0], forward_flow[1])
    warping_result = np.vstack((img1, img2))#, warped_img))
    plt.imshow(warping_result)
    plt.title("Warping from {} to {}".format(i, j))
    plt.show()

