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

import numpy as np
import torch
from tqdm import tqdm

from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler
from args import get_args
from threading import Thread
from dataloaders.image_dataset import ImageDataset
from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher

# CHANGE from poses.pose_initializer import PoseInitializer

from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.mono_depth import MonoDepthEstimator
from scene.scene_model import SceneModel
from gaussianviewer import GaussianViewer
from webviewer.webviewer import WebViewer
from graphdecoviewer.types import ViewerMode
from utils import align_mean_up_fwd, increment_runtime

# CHANGE START
import cv2
from streams.ids_stream import IDSStream
from streams.mocap_stream import MoCapStream
from streams.stream_matcher import StreamMatcher
# CHANGE END

if __name__ == "__main__":
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    args = get_args()
    args.lr_poses = 0
    args.lr_exposure = 0

    # Initialize dataloader

    # CHANGE START
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
    # CHANGE END

    height, width = dataset.get_image_size()

    # Initialize other modules
    print("Initializing modules and running just in time compilation, may take a while...")
    max_error = max(args.match_max_error * width, 1.5)
    min_displacement = max(args.min_displacement * width, 30)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_prev_keyframes_miniba_incr, max_error
    )

    # CHANGE START
    #pose_initializer = PoseInitializer(
    #    width, height, triangulator, matcher, 2 * max_error, args
    #)
    focal = 1.0 # dummy focal
    # CHANGE END

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

    n_active_keyframes = 0
    n_keyframes = 0
    needs_reboot = False
    bootstrap_keyframe_dicts = []
    bootstrap_desc_kpts = []

    # Dict of runtimes for each step
    runtimes = ["Load", "BAB", "tri", "BAI", "Add", "Init", "Opt", "anc"]
    runtimes = {key: [0, 0] for key in runtimes}
    metrics = {}

    ## Scene reconstruction
    print(f"Starting reconstruction for {args.source_path}")
    pbar = tqdm(range(0, len(dataset)))
    reconstruction_start_time = time.time()

    # CHANGE START
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
    # CHANGE END
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

            image, info = dataset.getnext()

            # CHANGE START
            if info is None:
                continue
            # CHANGE END

            if n_keyframes == 0:

                # CHANGE START
                if info is None:
                    continue
                # CHANGE END

                prev_desc_kpts = detector(image)
                bootstrap_keyframe_dicts = [{"image": image, "info": info}]
                bootstrap_desc_kpts = [prev_desc_kpts]
                n_keyframes += 1
                continue

            desc_kpts = detector(image)
            # Match features between the previous and current frame
            curr_prev_matches = matcher(desc_kpts, prev_desc_kpts)
            # Determine if we should add a keyframe based on the matches
            dist = torch.norm(curr_prev_matches.kpts - curr_prev_matches.kpts_other, dim=-1)
            should_add_keyframe = (
                dist.median() > min_displacement
                and len(curr_prev_matches.kpts) > args.min_num_inliers
            )
            # Always add test frames so we estimate their poses
            should_add_keyframe |= info["is_test"]
            increment_runtime(runtimes["Load"], start_time)
            
            # CHANGE START
            # Determine if we should add a keyframe based on its movement
            if is_stream:
                m_pos = info["m_pos"]
                m_rot = info["m_rot"]
                should_add_keyframe_movement = (m_pos < 0.5 and m_rot < 0.2)
                
                should_add_keyframe = (should_add_keyframe and should_add_keyframe_movement)
            # CHANGE END

            if should_add_keyframe:
                # CHANGE START
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
                # CHANGE END

                ## Bootstrap
                # Accumulate keyframes for pose initialization
                if n_keyframes < args.num_keyframes_miniba_bootstrap:
                    bootstrap_keyframe_dicts.append({"image": image, "info": info})
                    bootstrap_desc_kpts.append(desc_kpts)

                if n_keyframes == args.num_keyframes_miniba_bootstrap - 1:
                    start_time = time.time()

                    # CHANGE START
                    # Replace bootstrap initializing by just taking the Rts from the infos
                    Rts = torch.eye(4, device="cuda")[None].repeat(len(bootstrap_keyframe_dicts), 1, 1)
                    for i in range(1, len(Rts)):
                        Rts[i, :4, :4] = bootstrap_keyframe_dicts[i]["info"]["Rt"]
                    # Perform exhaustive matching
                    for i in range(len(bootstrap_keyframe_dicts)):
                        for j in range(i + 1, len(bootstrap_keyframe_dicts)):
                            _ = matcher(bootstrap_desc_kpts[i], bootstrap_desc_kpts[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
                    focal = info["focal"].item()
                    # CHANGE END

                    increment_runtime(runtimes["BAB"], start_time)
                    for index, (keyframe_dict, desc_kpts, Rt) in enumerate(
                        zip(bootstrap_keyframe_dicts, bootstrap_desc_kpts, Rts)
                    ):
                        start_time = time.time()
                        if args.use_colmap_poses:
                            Rt = keyframe_dict["info"]["Rt"]
                            f = keyframe_dict["info"]["focal"]
                        keyframe = Keyframe(
                            keyframe_dict["image"],
                            keyframe_dict["info"],
                            desc_kpts,
                            Rt,
                            index,
                            f,
                            dense_extractor,
                            depth_estimator,
                            triangulator,
                            args,
                        )
                        scene_model.add_keyframe(keyframe, f)
                        increment_runtime(runtimes["Add"], start_time)
                    if args.viewer_mode not in ["none", "web"]:
                        viewer.reset_intrinsics("point_view")
                    prev_keyframe = keyframe
                    for index in range(args.num_keyframes_miniba_bootstrap):
                        start_time = time.time()
                        scene_model.add_new_gaussians(index)
                        increment_runtime(runtimes["Init"], start_time)
                    start_time = time.time()
                    # Run initial optimization on the bootstrap keyframes
                    # If streaming, run async optimization until the next keyframe is added
                    if is_stream:
                        scene_model.optimize_async(args.num_iterations)
                    else:
                        scene_model.optimization_loop(args.num_iterations)
                    increment_runtime(runtimes["Opt"], start_time)
                    last_reboot = n_keyframes

                # CHANGE START
                """
                ## Reboot
                if (
                    args.enable_reboot
                    and scene_model.approx_cam_centres is not None
                    and len(scene_model.anchors)
                ):
                    # Check if the camera baseline is a lot smaller or larger than expected
                    last_centers = scene_model.approx_cam_centres[-20:]
                    rel_dist = torch.norm(
                        last_centers[1:] - last_centers[:-1], dim=-1
                    ).mean()
                    needs_reboot = (
                        rel_dist > 0.1 * 5 or rel_dist < 0.1 / 3
                    ) and n_keyframes - last_reboot > 50
                if needs_reboot:
                    # Reboot: run mini BA on the last 8 keyframes
                    bs_kfs = scene_model.keyframes[-8:]
                    bootstrap_desc_kpts = [bs_kf.desc_kpts for bs_kf in bs_kfs]
                    in_Rts = torch.stack([kf.get_Rt() for kf in bs_kfs])
                    Rts, _, final_residual = pose_initializer.initialize_bootstrap(
                        bootstrap_desc_kpts, rebooting=True
                    )
                    # Check if the reboot succeeded
                    if final_residual < max_error * 0.5:
                        Rts = align_mean_up_fwd(Rts, in_Rts)
                        for Rt, keyframe in zip(Rts, bs_kfs):
                            keyframe.set_Rt(Rt)
                        # Reset the scene model and reinitialize the gaussians
                        scene_model.reset()
                        for i in range(3, 0, -1):
                            scene_model.add_new_gaussians(-i)
                        for _ in range(3 * args.num_iterations):
                            scene_model.optimization_step()
                        needs_reboot = False
                        last_reboot = n_keyframes
                """
                # CHANGE END

                ## Incremental reconstruction
                # Incremental pose initialization
                if n_keyframes >= args.num_keyframes_miniba_bootstrap:
                    start_time = time.time()
                    prev_keyframes = scene_model.get_prev_keyframes(
                        args.num_prev_keyframes_miniba_incr, True, desc_kpts
                    )
                    increment_runtime(runtimes["tri"], start_time)
                    start_time = time.time()

                    # CHANGE START
                    Rt = info["Rt"]
                    for keyframe in prev_keyframes:
                        _ = matcher(desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=n_keyframes, kID_other=keyframe.index)
                    # CHANGE END

                    increment_runtime(runtimes["BAI"], start_time)
                    start_time = time.time()
                    if Rt is not None:
                        if args.use_colmap_poses:
                            Rt = info["Rt"]
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
                        scene_model.add_keyframe(keyframe)
                        prev_keyframe = keyframe
                        increment_runtime(runtimes["Add"], start_time)
                        # Gaussian initialization
                        start_time = time.time()
                        scene_model.add_new_gaussians()
                        increment_runtime(runtimes["Init"], start_time)
                        start_time = time.time()
                        # If streaming, run async optimization until the next keyframe is added
                        if is_stream:
                            scene_model.optimize_async(args.num_iterations)
                        else:
                            scene_model.optimization_loop(args.num_iterations)
                        increment_runtime(runtimes["Opt"], start_time)
                    else:
                        should_add_keyframe = False

            if should_add_keyframe:
                ## Check if anchor creation is needed based on the primitives' size 
                start_time = time.time()
                scene_model.place_anchor_if_needed()
                increment_runtime(runtimes["anc"], start_time)

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

                ## Display optimization progress and metrics
                bar_postfix = []
                for key, value in metrics.items():
                    bar_postfix += [f"\033[31m{key}:{value:.2f}\033[0m"]
                if args.display_runtimes:
                    for key, value in runtimes.items():
                        if value[1] > 0:
                            bar_postfix += [
                                f"\033[35m{key}:{1000 * value[0] / value[1]:.1f}\033[0m"
                            ]
                bar_postfix += [
                    f"\033[36mFocal:{focal:.1f}",
                    f"\033[36mKeyframes:{n_keyframes}\033[0m",
                    f"\033[36mGaussians:{scene_model.n_active_gaussians}\033[0m",
                    f"\033[36mAnchors:{len(scene_model.anchors)}\033[0m",
                ]
                pbar.set_postfix_str(",".join(bar_postfix), refresh=False)

    # CHANGE START
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping reconstruction...")
        pbar.close()
    # CHANGE END

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

    # CHANGE START
    print("Cleaning up threads...")
    scene_model.join_optimization_thread()
    if is_stream:
        cam_stream.stop()
        mocap_stream.stop()
        poses_file.close()
    # CHANGE END