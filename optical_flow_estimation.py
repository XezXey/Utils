# Author: Deepak Pathak (c) 2016
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import sys
import os
from tqdm import tqdm
sys.path.append('/home/puntawat/Mint/Work/Vision/3D_Human_Reconstruction/pyflow')
import pyflow
import matplotlib.pyplot as plt
import cv2
import multiprocessing

# Flow Options:
alpha = 0.03 # Smoothness of frame (Higher means lower movement/difference capture => small optical flow offset)
ratio = 0.95 # Decreasing rate of the image size per each pyramid
minWidth = 5 # Size of receptive field for each image (Lower is better to capture a large movement)
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def compute_flow(i, im1, im2, normalize, output_optical_flow_path_fw, output_optical_flow_path_bw):
  # Normalization flag for input images
  if normalize:
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) /255.
  # Calculate a forward sequence optical flow 
  u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
  flows_forward = np.concatenate((u[..., None], v[..., None]), axis=2)
  # Create a Forward frame
  cv2.imwrite(output_optical_flow_path_fw + 'outflow_warped_frame{}.png'.format(i), im2W[:, :, ::-1] * 255)

  # Calculate a backward sequence optical flow 
  u, v, im2W = pyflow.coarse2fine_flow(
    im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
  flows_backward = np.concatenate((u[..., None], v[..., None]), axis=2)
  # Create a Backward frame
  cv2.imwrite(output_optical_flow_path_bw + 'outflow_warped_frame{}.png'.format(i), im2W[:, :, ::-1] * 255)
  print("SHAPE : ", flows_forward.shape, flows_backward.shape)
  return flows_forward, flows_backward

def optical_flow_estimation(images, output_path='./', normalize=False):
  # Create an optical flow folder for saving a results
  output_optical_flow_path = output_path[:-11] + "OpticalFlowEstimation/"
  output_optical_flow_path_fw = output_optical_flow_path + "Forward/"
  output_optical_flow_path_bw = output_optical_flow_path + "Backward/"
  if not os.path.exists(output_optical_flow_path):
    os.mkdir(output_optical_flow_path)
  if not os.path.exists(output_optical_flow_path_fw):
    os.mkdir(output_optical_flow_path_fw)
  if not os.path.exists(output_optical_flow_path_bw):
    os.mkdir(output_optical_flow_path_bw)
  flows_forward = []
  flows_backward = []
  pooling = multiprocessing.Pool(multiprocessing.cpu_count())
  print("Optical Flow Estimation ...")
  # Using multiprocessing pool to compute flows in parallel
  flows_fw_bw = pooling.starmap(compute_flow, (zip(list(range(len(images)-1)),
                                            [images[i] for i in range(len(images)-1)],
                                            [images[i+1] for i in range(len(images)-1)],
                                            [normalize] * (len(images)-1),
                                            [output_optical_flow_path_fw] * (len(images)-1),
                                            [output_optical_flow_path_bw] * (len(images)-1),)))
  flows_fw_bw = np.array(flows_fw_bw)
  # Split forward and backward from flows_fw_bw (#N_frame, 2(forward and backward), h, w, 2(u and v))
  flows_forward = np.array(flows_fw_bw[:, 0, ...])
  flows_backward = np.array(flows_fw_bw[:, 1, ...])
  # Summary and save to the output path
  print("OPTICAL FLOW SUMMARY : ")
  print("Forward Sequence FLOWS : ", flows_backward.shape)
  print("Backward Sequence FLOWS : ", flows_forward.shape)
  print("Saving optical_flow_estimation to : ", output_optical_flow_path)
  np.save(output_optical_flow_path + "flows_forward.npy", flows_forward)
  np.save(output_optical_flow_path + "flows_backward.npy", flows_backward)
  return flows_forward, flows_backward

def refinement_kps(kps, images, optical_flow_forward, optical_flow_backward, sequence_length=14, refinement_method='mean'):
  print("Keypoints : ", kps.shape)
  print("Optical Flow Forward : ", optical_flow_forward.shape)
  print("Images size : ", np.array(list(images)).shape)
  # Convert kps back to image space
  kps = ((kps+1) * 0.5 * images[0].shape[0])
  neighbour_length = sequence_length // 2 # Define a number of look forward/backward frame.
  # Forward lookup use a backward optical flow
  print("[*]Performing a forward lookup")
  refined_kps_forward_lookup_all_frames = get_refined_kps_forward(kps=kps, images=images, neighbour_length=neighbour_length, optical_flow=optical_flow_backward)
  # Backward lookup use a forwardward optical flow
  print("[*]Performing a backward lookup")
  refined_kps_backward_lookup_all_frames = get_refined_kps_backward(kps=kps, images=images, neighbour_length=neighbour_length, optical_flow=optical_flow_forward)
  print("Refined Keypoints backward lookup : ", np.array(refined_kps_backward_lookup_all_frames)[0].shape)
  print("Refined Keypoints forward lookup : ", np.array(refined_kps_forward_lookup_all_frames)[0].shape)
  voted_keypoints = keypoints_voting(refined_kps_forward_lookup=np.array(refined_kps_forward_lookup_all_frames), refined_kps_backward_lookup=np.array(refined_kps_backward_lookup_all_frames), images=images, voting_method=refinement_method)
  return voted_keypoints


def get_refined_kps_forward(kps, images, neighbour_length, optical_flow):
  imsize = optical_flow.shape[1]
  optical_flow_backward = optical_flow
  refined_kps_all_frames = []
  kps = np.clip(kps[...], 0, imsize-1)
  for i in range(len(images)): # Perform frame-by-frame
    refined_kps = []
    print("[*]Keypoints at {} frame".format(i))
    print("[*]Image at {} frame".format(i))
    # Perform backward lookup
    if i + neighbour_length > len(images)-1:
      forward_lookup = len(images)-1-i # Use only #n frame left in the sequence
    else :
      forward_lookup = neighbour_length
    print("--->Avaialbe Images to look up : ", forward_lookup)
    for j in range(i + forward_lookup, i, -1): # Iterate to get the voters
      print("--->Available Keypoints : ", kps[i:j, :, :].shape)
      print("--->Available Optical Flow :", optical_flow_backward[i:j, kps[i:j, :, 0].astype(int), kps[i:j, :, 1].astype(int), :].shape)
      # Initialize the Combine flow by start it with the first frame(first in neighbour sequence)
      # Should perform a bilinear interpolation first
      flow_from_img_j_to_i_x = kps[j, :, 0]
      flow_from_img_j_to_i_y = kps[j, :, 1]
      for combine_flow_step in range(j-1, i-1, -1):
        # print("===>COMBINING STEP at frame {} (Back To frame {} )".format(combine_flow_step, i))
        # print("===>FLOW(X) : ", flow_from_img_j_to_i_x, "\nADDING WITH : ", optical_flow_backward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 0])
        # print("===>FLOW(Y) : ", flow_from_img_j_to_i_y, "\nADDING WITH : ", optical_flow_backward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 1])
        # Adding a keypoints with flow to next frame
        flow_from_img_j_to_i_x_indexing = np.clip(flow_from_img_j_to_i_x.astype(int), 0, imsize-1)
        flow_from_img_j_to_i_y_indexing = np.clip(flow_from_img_j_to_i_y.astype(int), 0, imsize-1)
        flow_from_img_j_to_i_x = flow_from_img_j_to_i_x + optical_flow_backward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 0]
        flow_from_img_j_to_i_y = flow_from_img_j_to_i_y + optical_flow_backward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 1]
        # This will change the flow variable type back to float type ===> So we need to cast it back to integer to use as an index
        # Can be replace with bilinear-interpolation
        flow_from_img_j_to_i_x_indexing = flow_from_img_j_to_i_x.astype(int)
        flow_from_img_j_to_i_y_indexing = flow_from_img_j_to_i_y.astype(int)
      # print("=" * 100)
      # print("The final flow results from frame {} to {}".format(i, j))
      # print("Flow(x) : ", np.array(flow_from_img_j_to_i_x).shape, "\nValue : ", flow_from_img_j_to_i_x)
      # print("Flow(y) : ",  np.array(flow_from_img_j_to_i_y).shape, "\nValue : ", flow_from_img_j_to_i_y)
      # print("=" * 100)
      refined_kps.append([flow_from_img_j_to_i_x, flow_from_img_j_to_i_y])
    refined_kps_all_frames.append(np.array(refined_kps))
    print("Summary refined kps at image {} : ".format(i), np.array(refined_kps).shape)
    print("***If the optical flow add up with keypoints is work correctly, the results from each refined_kps given each frame should stay closely together")
    print("=" * 100)
  print("Summary refined kps from all images : ", np.array(refined_kps_all_frames).shape)
  # print("Summary refined kps from all images : ", np.array(refined_kps_all_frames))
  return refined_kps_all_frames

def get_refined_kps_backward(kps, images, neighbour_length, optical_flow):
  imsize = optical_flow.shape[1]
  kps = np.clip(kps[...], 0, imsize-1)
  optical_flow_forward = optical_flow
  refined_kps_all_frames = []
  for i in range(len(images)): # Perform frame-by-frame
    refined_kps = []
    print("[*]Keypoints at {} frame".format(i))
    print("[*]Image at {} frame".format(i))
    # Perform backward lookup
    if i - neighbour_length < 0:
      backward_lookup = i # Use only #n frame left in the sequence
    else :
      backward_lookup = neighbour_length

    print("--->Avaialbe Images to look up : ", backward_lookup)
    for j in range(i - backward_lookup, i): # Iterate to get the voters
      print("--->Available Keypoints : ", kps[j:i, :, :].shape)
      print("--->Available Optical Flow :", optical_flow_forward[j:i, kps[j:i, :, 0].astype(int), kps[j:i, :, 1].astype(int), :].shape)
      # Initialize the Combine flow by start it with the first frame(first in neighbour sequence)
      # Should perform a bilinear interpolation first
      flow_from_img_j_to_i_x = kps[j, :, 0]
      flow_from_img_j_to_i_y = kps[j, :, 1]
      for combine_flow_step in range(j, i):
        # print("===>COMBINING STEP at frame {} (To {} frame)".format(combine_flow_step, i))
        # print("===>FLOW(X) : ", flow_from_img_j_to_i_x, "\nADDING WITH : ", optical_flow_forward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 0])
        # print("===>FLOW(Y) : ", flow_from_img_j_to_i_y, "\nADDING WITH : ", optical_flow_forward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 1])
        # Adding a keypoints with flow to next frame
        flow_from_img_j_to_i_x_indexing = np.clip(flow_from_img_j_to_i_x.astype(int), 0, imsize-1)
        flow_from_img_j_to_i_y_indexing = np.clip(flow_from_img_j_to_i_y.astype(int), 0, imsize-1)
        flow_from_img_j_to_i_x = flow_from_img_j_to_i_x + optical_flow_forward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 0]
        flow_from_img_j_to_i_y = flow_from_img_j_to_i_y + optical_flow_forward[combine_flow_step, flow_from_img_j_to_i_x_indexing, flow_from_img_j_to_i_y_indexing, 1]
        # This will change the flow variable type back to float type ===> So we need to cast it back to integer to use as an index
        # Can be replace with bilinear-interpolation
        flow_from_img_j_to_i_x_indexing = flow_from_img_j_to_i_x.astype(int)
        flow_from_img_j_to_i_y_indexing = flow_from_img_j_to_i_y.astype(int)
      # print("=" * 100)
      # print("The final flow results from frame {} to {}".format(j, i))
      # print("Flow(x) : ", np.array(flow_from_img_j_to_i_x).shape, "\nValue : ", flow_from_img_j_to_i_x)
      # print("Flow(y) : ",  np.array(flow_from_img_j_to_i_y).shape, "\nValue : ", flow_from_img_j_to_i_y)
      print("=" * 100)
      refined_kps.append([flow_from_img_j_to_i_x, flow_from_img_j_to_i_y])
    refined_kps_all_frames.append(np.array(refined_kps))
    print("Summary refined kps at image {} : ".format(i), np.array(refined_kps).shape)
    print("***If the optical flow add up with keypoints is work correctly, the results from each refined_kps given each frame should stay closely together")
    print("=" * 100)
  print("Summary refined kps from all images : ", np.array(refined_kps_all_frames).shape)
  # print("Summary refined kps from all images : ", np.array(refined_kps_all_frames))
  return refined_kps_all_frames

def keypoints_voting(refined_kps_forward_lookup, refined_kps_backward_lookup, images, voting_method='mean'):
  all_frame_voted = []
  for frame_idx in range(len(images)):
    if refined_kps_forward_lookup[frame_idx].shape[0] == 0:
      combine_refined_kps = refined_kps_backward_lookup[frame_idx]
    elif refined_kps_backward_lookup[frame_idx].shape[0] == 0:
      combine_refined_kps = refined_kps_forward_lookup[frame_idx]
    else:
      combine_refined_kps = np.concatenate((refined_kps_forward_lookup[frame_idx], refined_kps_backward_lookup[frame_idx]))

    print("At frame {} : ".format(frame_idx), "Combined : ", combine_refined_kps.shape)
    # print("===>Forward Lookup : ", refined_kps_forward_lookup[frame_idx])#, refined_kps_forward_lookup[frame_idx])
    # print("===>Backward Lookup : ", refined_kps_backward_lookup[frame_idx])#, refined_kps_backward_lookup[frame_idx])
    # print(combine_refined_kps)
    print("===>Forward Lookup : ", refined_kps_forward_lookup[frame_idx].shape)#, refined_kps_forward_lookup[frame_idx])
    print("===>Backward Lookup : ", refined_kps_backward_lookup[frame_idx].shape)#, refined_kps_backward_lookup[frame_idx])
    print("Using {} voting : ", np.mean(combine_refined_kps, axis=0).T.shape)
    # Performing a voting method
    if voting_method == 'mean':
      all_frame_voted.append(np.mean(combine_refined_kps, axis=0).T)
    elif voting_method == 'median':
      all_frame_voted.append(np.median(combine_refined_kps, axis=0).T)
  all_frame_voted = np.array(all_frame_voted)
  print("The finale refinement kps using Optical Flow (The shape should be (#n_frame, 2(x, y), 25(#n_kps)) : ", all_frame_voted.shape)
  return all_frame_voted


def bilinear_interpolation():

  pass

