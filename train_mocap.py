#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import shutil

import numpy as np
import torch
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
import cv2

if __name__ == "__main__":
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    args = get_args()

    #args.lr_poses = 0
    #args.lr_exposure = 0
    
    if args.source_path == "ids":
        cam_stream = IDSStream(frame_rate=45, 
                                exposure_time=20000, 
                                white_balance='auto',
                                gain='auto',
                                gamma=1.5)

        mocap_stream = MoCapStream(client_ip="172.22.147.168", # 168 for workstation, 172 for laptop
                                    server_ip="172.22.147.182", 
                                    buffer_size=20)

        dataset = StreamMatcher(cam_stream, mocap_stream, rb_id=2, calib_path="latest", undistort=True, downsampling=3)
        is_stream = True

    else:
        dataset = ImageDataset(args)
        is_stream = False

    height, width = dataset.get_image_size()

    # Initialize other modules
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)

    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )
    dense_extractor = DenseExtractor(width, height)
    depth_estimator = MonoDepthEstimator(width, height)
    scene_model = SceneModel(width, height, args, matcher)
    detector = Detector(args.num_kpts, width, height)

    # Initialize the viewer
    if args.viewer_mode in ["server", "local"]:
        viewer_mode = ViewerMode.SERVER if args.viewer_mode == "server" else ViewerMode.LOCAL
        viewer = GaussianViewer.from_scene_model(scene_model, viewer_mode)
        viewer_thd = Thread(target=viewer.run, args=(args.ip, args.port), daemon=True)
        viewer_thd.start()
        viewer.throttling = True # Enable throttling when training
    elif args.viewer_mode == "web":
        ip = "0.0.0.0"
        server = TCPServer((ip, 8000), SimpleHTTPRequestHandler)
        server_thd = Thread(target=server.serve_forever, daemon=True)
        server_thd.start()
        print(f"Visit http://{ip}:8000/webviewer to for the viewer")

        viewer = WebViewer(scene_model, args.ip, args.port)
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

    ## Scene reconstruction
    print(f"Starting reconstruction for {args.source_path}")
    pbar = tqdm(range(0, len(dataset)))
    reconstruction_start_time = time.time()

    n_keyframes = 0
    bootstrap_keyframes = []
    min_displacement = max(args.min_displacement * width, 30)
    metrics = {}

    # If capturing live: Save intrinsics, captured images and poses to model_path
    if is_stream:
        images_dir = os.path.join(args.model_path, "images")
        os.makedirs(images_dir)

        poses_dir = os.path.join(args.model_path, "sparse", "0")
        os.makedirs(poses_dir)

        points3D_path = os.path.join(poses_dir, "points3D.txt") # Dummy file
        cameras_path = os.path.join(poses_dir, "cameras.txt")
        poses_path = os.path.join(poses_dir, "images.txt")

        with open(points3D_path, "w") as f:
            pass

        image, info = dataset.getnext()
        camera_matrix = info["camera_matrix"]
        focal = info["focal"].item()
        with open(cameras_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, w, h, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            f.write("# PARAMS for PINHOLE are: w, h, fx, fy, cx, cy\n")
            f.write(f"1 PINHOLE {width} {height} {focal:.6f} {focal:.6f} {width/2:.6f} {height/2:.6f}\n")

        poses_file = open(poses_path, "w")
        poses_file.write("# Image list with two lines of data per image:\n")
        poses_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        poses_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        poses_file.write("# Number of images: PLACEHOLDER, mean observations per image: 0\n")
        poses_file.write("# These poses have been captured with a MoCap system\n")

    try:
        for frameID in pbar:
            start_time = time.time()

            if args.viewer_mode == "web":
                viewer.trainer_state = "running"

                # Paused
                while viewer.state == "stop":
                    pbar.set_postfix_str(
                        "\033[31mPaused. Press the Start button in the webviewer\033[0m"
                    )
                    time.sleep(0.1)
                
                # Finish reconstruction
                if viewer.state == "finish":
                    viewer.trainer_state = "finish"
                    break
            
            # Get data
            image, info = dataset.getnext()

            if info is None:
                continue

            if is_stream:
                # Save image and pose data
                image_name = f"{n_keyframes:04d}.png"
                image_save = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(images_dir, image_name), image_save)
                pos = info["pos"]
                rot = info["rot"]
                poses_file.write(
                    f"{n_keyframes} {rot[3]:.6f} {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} 1 {image_name}\n\n"
                )
                poses_file.flush()

            # Extract Rt, f and compute keypoints          
            Rt = info["Rt"]
            f = info["focal"]
            focal = f.item()
            desc_kpts = detector(image)

            if n_keyframes == 0:
                prev_desc_kpts = detector(image)
                first_keyframe = Keyframe(
                    image,
                    info,
                    prev_desc_kpts,
                    Rt,
                    n_keyframes,
                    f,
                    dense_extractor,
                    depth_estimator,
                    triangulator,
                    args,
                )
                bootstrap_keyframes.append(first_keyframe)
                n_keyframes += 1
                continue

            # Match features between previous and current frame
            curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)

            # Determine if we should add a keyframe based on the matches
            dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
            should_add_keyframe_matches = (
                dist.median() > min_displacement
                and len(curr_prev_matches.kpts) > args.min_num_inliers
            )

            if not should_add_keyframe_matches:
                continue
            
            if is_stream:
                # Determine if we should add a keyframe based on its velocity
                m_pos = info["m_pos"]
                m_rot = info["m_rot"]
                should_add_keyframe_velocity = (m_pos < 0.5 and m_rot < 0.2)
                
                if not should_add_keyframe_velocity:
                    continue
            
            # Update matches in desc_kpts
            desc_kpts.update_matches(n_keyframes - 1, curr_prev_matches)
            prev_desc_kpts.update_matches(n_keyframes, curr_prev_matches, swap=True)

            # Create Keyframe object
            keyframe = Keyframe(
                image,
                info,
                desc_kpts,
                Rt,
                n_keyframes,
                f,
                dense_extractor,
                depth_estimator,
                triangulator,
                args,
            )

            # Wait until 8 keyframes are in the scene_model
            required_kfs = 8
            if n_keyframes + 1 < required_kfs:
                bootstrap_keyframes.append(keyframe)

            # When there are 8 keyframes in the scene_model, add all gaussians at once and start optimizing
            elif n_keyframes + 1 == required_kfs:
                # Add all bootstrap keyframes first
                for bootstrap_kf in bootstrap_keyframes:
                    scene_model.add_keyframe(bootstrap_kf, f)
                # Add the current keyframe as well
                scene_model.add_keyframe(keyframe, f)
                # Now add gaussians for all keyframes
                for index in range(required_kfs):
                    scene_model.add_new_gaussians(index)
                # Now optimize
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                elif not is_stream:
                    scene_model.optimization_loop(args.num_iterations)

            # For every new keyframe: add keyframe and gaussians to the scene_model
            elif n_keyframes + 1 > required_kfs:
                scene_model.add_keyframe(keyframe, f)
                prev_keyframes = scene_model.get_prev_keyframes(n=required_kfs, update_3dpts=True, desc_kpts=desc_kpts) # makes sure "update_3dpts" is executed
                for prev_keyframe in prev_keyframes: # builds matches in the last 8 keyframes and removes outliers
                    matches = matcher(desc_kpts, prev_keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=n_keyframes, kID_other=prev_keyframe.index)
                scene_model.add_new_gaussians()
                if is_stream:
                    scene_model.optimize_async(args.num_iterations)
                elif not is_stream:
                    scene_model.optimization_loop(args.num_iterations)

            ## Check if anchor creation is needed based on the primitives' size 
            scene_model.place_anchor_if_needed()

            n_keyframes += 1
            if not info["is_test"]:
                prev_desc_kpts = desc_kpts

            ## Intermediate evaluation
            if (
                n_keyframes % args.test_frequency == 0
                and args.test_frequency > 0
                and (args.test_hold > 0 or args.eval_poses)
            ):
                metrics = scene_model.evaluate(args.eval_poses)

            ## Save intermediate model
            if (
                frameID % args.save_every == 0
                and args.save_every > 0
            ):
                scene_model.save(
                    os.path.join(args.model_path, "progress", f"{frameID:05d}")
                )
                input("Ready for sync?")
                print(dataset.get_time_diff())

            ## Display optimization progress and metrics
            bar_postfix = []
            for key, value in metrics.items():
                bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
            bar_postfix += [
                f"\033[36mFocal:{focal:.1f}",
                f"\033[36mKeyframes:{n_keyframes}\033[0m",
                f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
            ]
            pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping reconstruction...")
        pbar.close()

    reconstruction_time = time.time() - reconstruction_start_time

    # Set to inference mode so that the model can be rendered properly
    scene_model.enable_inference_mode()

    # Save the model and metrics
    print("Saving the reconstruction to:", args.model_path)
    metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
    print(
        ", ".join(
            f"{metric}: {value:.3f}"
            if isinstance(value, float)
            else f"{metric}: {value}"
            for metric, value in metrics.items()
        )
    )

    # Fine tuning after initial reconstruction
    if len(args.save_at_finetune_epoch) > 0:
        finetune_epochs = max(args.save_at_finetune_epoch)
        torch.cuda.empty_cache()
        scene_model.inference_mode = False
        pbar = tqdm(range(0, finetune_epochs), desc="Fine tuning")
        for epoch in pbar:
            # Run one epoch of fine-tuning
            epoch_start_time = time.time()
            scene_model.finetune_epoch()
            epoch_time = time.time() - epoch_start_time
            reconstruction_time += epoch_time
            # Save the model and metrics
            if epoch + 1 in args.save_at_finetune_epoch:
                torch.cuda.empty_cache()
                scene_model.inference_mode = True
                metrics = scene_model.save(
                    os.path.join(args.model_path, str(epoch + 1)), reconstruction_time
                )
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                pbar.set_postfix_str(",".join(bar_postfix))
                scene_model.inference_mode = False
                torch.cuda.empty_cache()
                
        # Set to inference mode so that the model can be rendered properly
        scene_model.inference_mode = True

    if args.viewer_mode != "none":
        if args.viewer_mode == "web":
            while True:
                time.sleep(1)
        else:
            viewer.throttling = False # Disable throttling when done training
            # Loop to keep the viewer alive
            while viewer.running:
                time.sleep(1)
    
    # Clean up all threads before exit
    print("Cleaning up threads...")
    scene_model.join_optimization_thread()
    if is_stream:
        cam_stream.stop()
        mocap_stream.stop()
        poses_file.close()


