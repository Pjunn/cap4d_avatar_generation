from typing import Dict, List, Tuple
from pathlib import Path
import argparse
import shutil
import json

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import roma
import trimesh
from scipy.ndimage import gaussian_filter1d

from flowface.flame.utils import (
    OPENCV2PYTORCH3D, 
    transform_vertices,
)
from cap4d.flame.flame import CAP4DFlameSkinner
from cap4d.datasets.utils import pivot_camera_intrinsic



ORBIT_PERIOD = 4 #8  # orbit period in seconds
ORBIT_AMPLITUDE_YAW = 15 #30 #55  # yaw angle amplitude for orbit
ORBIT_AMPLITUDE_PITCH = 5 #20  # pitch angle amplitude for orbit
MAX_EYE_ROTATION = 25  # maximum eyeball rotation angle in degrees


class FlameFittingModel(nn.Module):
    def __init__(
        self,
        flame: CAP4DFlameSkinner,
        n_timesteps: int,
        vertex_weights: torch.Tensor,
        use_jaw_rotation: bool = False,
    ):
        """
        flame: the flame model to use (Note: vertex mask needs to be applied to this model)
        cam_resolutions: resolutions of the cameras
        n_timesteps: number of timesteps
        n_cams: number of cameras
        fps: frames per second of the sequence for acceleration loss calculation - if fps < 0, no acceleration loss
        use_jaw_rotation: whether or not to use jaw rotation (make sure the appropriate flame model is loaded)
        camera_calibration: camera calibration dictionary containing
            "intrinsics": torch.Tensor with (N_c, 3, 3) intrinsic matrices (K)
            "extrinsics": torch.Tensor with (N_c, 4, 4) extrinsic transforms (RT)
        vertex_mask: torch.Tensor with (N_v) indicating the vertices used for tracking
        """
        super().__init__()

        n_expr_params = flame.n_expr_params
        n_shape_params = flame.n_shape_params
        self.use_jaw_rotation = use_jaw_rotation
        self.n_timesteps = n_timesteps

        self.flame = flame
        self.flame = torch.compile(self.flame, mode="reduce-overhead")
        # initialize flame sequence parameters
        self.shape = nn.Parameter(torch.zeros(n_shape_params))
        self.expr = nn.Parameter(torch.zeros(n_timesteps, n_expr_params))
        self.rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.tra = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.eye_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.neck_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        if use_jaw_rotation:
            self.jaw_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
            # some heuristic normal deviations of jaw rotations for loss calculation
            self.register_buffer("jaw_std", torch.deg2rad(torch.tensor([45, 5, 0.01])), persistent=False)
        else:
            self.jaw_rot = None
        
        # utils
        # self.register_buffer("vertex_mask", vertex_mask, persistent=False)
        self.register_buffer("vertex_weights", vertex_weights / vertex_weights.sum(), persistent=False)
        self.register_buffer("opencv2pytorch", OPENCV2PYTORCH3D, persistent=False)
    
    def _compute_reg_losses(self, verts_3d: torch.Tensor):
        l_shape = (self.shape ** 2).sum(dim=-1).mean()

        expr_params = self.expr
        if self.use_jaw_rotation:
            # normalize jaw rotation values
            jaw_values = self.jaw_rot / self.jaw_std[None]
            expr_params = torch.cat([expr_params, jaw_values], dim=-1)

        l_expr = (expr_params ** 2).sum(dim=-1).mean()

        return {
            "l_shape": l_shape,
            "l_expr": l_expr,
        }

    def forward(self):
        flame_sequence = {
            "shape": self.shape,
            "expr": self.expr,
            "rot": self.rot,
            "tra": self.tra,
            "eye_rot": self.eye_rot,
            "jaw_rot": self.jaw_rot,
            "neck_rot": self.neck_rot,
        }

        # compute FLAME vertices
        verts_3d, _ = self.flame(
            flame_sequence, 
        )

        # transform into OpenCV camera coordinate convention
        verts_3d_cv = transform_vertices(self.opencv2pytorch[None], verts_3d)  # [N_t V 3]

        return {
            "verts_3d": verts_3d_cv,
        }

    def fit(
        self,
        verts_3d: torch.Tensor,
        init_lr: float = 0.01, #0.05, #1e-2, #TODO
        n_steps: int = 6000,
        w_shape_reg: float = 1e-2,
        w_expr_reg: float = 1e-2,
        verbose: bool = True,
        pos_warm_up_steps: int = 500,
    ):
        """
        Fit 3D FLAME model to 2D alignment computed by the alignment module
        init_lr: initial learning rate
        n_steps: number of fitting steps
        w_shape_reg: FLAME shape parameter regularization
        w_expr_reg: FLAME expression parameter regularization
        """

        opt = torch.optim.Adam(
            lr=init_lr, params=self.parameters(), betas=(0.96, 0.999), 
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=25, factor=0.5 # TODO
        )

        if verbose:
            pbar = tqdm(range(n_steps))
        else:
            pbar = range(n_steps)
        for i in pbar:
            if i < pos_warm_up_steps:
                self.expr.data *= 0.
                self.shape.data *= 0.
                self.eye_rot.data *= 0.
            # self.neck_rot.data = torch.zeros_like(self.neck_rot.data)  # * 0.5

            opt.zero_grad(set_to_none=True)

            output_dict = self.forward()

            # compute 2D alignment loss
            l_vert = (output_dict["verts_3d"] - verts_3d) / 0.01  # [N_t V 3] # 만약 이런거면 3d alignment 라고 할수 있음. 
            l_vert = l_vert.norm(dim=-1)
            l_vert_max = l_vert.max()
            l_vert = l_vert ** 2 ## SQURE HACK
            l_vert = (l_vert * self.vertex_weights[None]).sum(dim=-1)
            l_vert = l_vert.mean()  # apply valid view mask

            reg_dict = self._compute_reg_losses(output_dict["verts_3d"])

            loss = l_vert
            loss += reg_dict["l_shape"] * w_shape_reg
            loss += reg_dict["l_expr"] * w_expr_reg

            loss = loss.mean()
            loss.backward()

            opt.step()

            if i > pos_warm_up_steps:
                scheduler.step(loss.item())

            if opt.param_groups[0]["lr"] < 1e-5:
                break

            if i % 10 == 0 and verbose:
                desc = ""
                desc += f"lr: {opt.param_groups[0]['lr']}, "
                desc += f"l: {loss.item():.3f}, "
                desc += f"vert: {l_vert.item():.3f}, "
                desc += f"vert_max: {l_vert_max.item():.3f}, "
                desc += f"shp: {reg_dict['l_shape'].item():.2f}, "
                desc += f"expr: {reg_dict['l_expr'].item():.2f}, "
                desc += f"shape_max: {self.shape.max().item():.2f}, "
                desc += f"expr_max: {self.expr.max().item():.2f}, "
                pbar.set_description(desc)

        return l_vert, output_dict["verts_3d"]

    def export_results(self):
        """
        Return the fitted parameters.
        """

        fit_3d = {
            "shape": self.shape.data.detach().cpu().numpy(),
            "expr": self.expr.data.detach().cpu().numpy(),
            "rot": self.rot.data.detach().cpu().numpy(),
            "tra": self.tra.data.detach().cpu().numpy(),
            "eye_rot": self.eye_rot.data.detach().cpu().numpy(),
            "neck_rot": self.neck_rot.data.detach().cpu().numpy(),
        }
        if self.jaw_rot is not None:
            fit_3d["jaw_rot"] = self.jaw_rot.data.detach().cpu().numpy()
        
        return fit_3d


def fit_flame(
    verts_3d, 
    gaze_directions,
    cam_rt,
    use_jaw_rotation=False,
    n_shape_params=150, 
    n_expr_params=65, 
    device="cpu",
    n_steps: int = 6000,
    w_shape_reg: float = 1e-2,
    w_expr_reg: float = 1e-2,
    smooth_eye_rotations: bool = False,
):
    verts_3d = torch.tensor(verts_3d).float().to(device)

    # vert_mask = torch.tensor(np.load("data/assets/flame/flowface_vertex_mask.npy"))
    vert_weights = torch.tensor(np.load("data/assets/flame/flowface_vertex_weights.npy"))
    # vert_weights = torch.ones_like(vert_weights)
    # vert_weights = vert_weights * vert_mask
    # import pdb; pdb.set_trace()

    flame_path = "data/assets/flame/flame2023_no_jaw.pkl"
    if use_jaw_rotation:
        flame_path = "data/assets/flame/flame2023.pkl"

    flame = CAP4DFlameSkinner(
        flame_path, 
        n_shape_params=n_shape_params, 
        n_expr_params=n_expr_params, 
        blink_blendshape_path="data/assets/flame/blink_blendshape.npy",
    ).to(device)

    fitter = FlameFittingModel(
        flame, 
        verts_3d.shape[0], 
        vertex_weights=vert_weights, 
        use_jaw_rotation=use_jaw_rotation
    ).to(device)
    _, pred_verts_3d = fitter.fit(
        verts_3d,
        n_steps=n_steps,
        w_shape_reg=w_shape_reg,
        w_expr_reg=w_expr_reg,
    )

    # fix eye rotations
    # eye rot 은 fittin 의 대상이 아님.
    for frame_id in range(verts_3d.shape[0]):
        if gaze_directions is None: # 입력 안들어 오면 상관 없음.
            eye_rot = np.zeros(3)
        else:
            ...
            
        fitter.eye_rot.data[frame_id] = torch.from_numpy(eye_rot).float().to(device)

    clamp_factor = fitter.eye_rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fitter.eye_rot.data = fitter.eye_rot.data / clamp_factor * clamp_factor.clamp(max=1.)

    if smooth_eye_rotations:
        fitter.eye_rot.data = torch.from_numpy(gaussian_filter1d(
            fitter.eye_rot.data.cpu().numpy(), sigma=2, axis=0
        )).float().to(device)

    # rerun flame
    pred_verts_3d = fitter.forward()["verts_3d"]

    return (
        fitter.export_results(), 
        pred_verts_3d.detach().cpu().numpy(), 
        flame.template_faces.cpu().numpy(),
    )


def convert_calibration(
    tracking_resolution,
    crop_box,
    k, 
):
    """
    Convert camera intrinsics from cropped and resized space to original resolution space.

    Parameters:
    orig_resolution (tuple): (H_orig, W_orig)
    tracking_resolution (tuple): (H_track, W_track)
    crop_box (tuple): (x0, y0, crop_w, crop_h) in original resolution
    rt (np.ndarray): [4, 4] extrinsic matrix (unchanged here)
    k (np.ndarray): [3, 3] intrinsics in tracking resolution

    Returns:
    new_k (np.ndarray): [3, 3] intrinsics in original resolution
    rt (np.ndarray): unchanged extrinsics
    """
    x0, y0, x1, y1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0
    H_track, W_track = tracking_resolution

    # 1. Compute the scale factor used for resizing the cropped image to tracking resolution
    scale_x = crop_w / W_track
    scale_y = crop_h / H_track

    # 2. Undo the scale (tracking → crop)
    k[0, :] *= scale_x  # fx, cx
    k[1, :] *= scale_y  # fy, cy

    # 3. Undo the crop (crop → original)
    k[0, 2] += x0  # cx
    k[1, 2] += y0  # cy

    return k


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def main(args):
    verts_3d = np.load(args.input_path) # (n_T, 5023, 3)
    
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # 시작. 
    print("Converting ARTalk results to FlowFace format.")

    print(f"Output path: {output_path}")
    print(f"Number of frames: {verts_3d.shape[0]}")


    out_flame = {}
    k_converted = np.array([[2604.74947368, 0, 257.33256217],[0,2604.74947368,269.5092192],[0,0,0]])
    rt = np.eye(4)
    orig_resolution = [512,512]
    # write camera calibration
    # fitting 에 사용안됨. . 
    out_flame["fx"] = k_converted[0, 0][None, None].astype(np.float32)
    out_flame["fy"] = k_converted[1, 1][None, None].astype(np.float32)
    out_flame["cx"] = k_converted[0, 2][None, None].astype(np.float32)
    out_flame["cy"] = k_converted[1, 2][None, None].astype(np.float32)
    out_flame["extr"] = rt[None].astype(np.float32)
    out_flame["resolutions"] = np.array([orig_resolution])
    out_flame["camera_order"] = ["cam0"]

    # mesh 불러와서 변환한거 convert 입력으로 사용.
    # verts_3d = np.stack(all_vertices, axis=0) # (frame, 5023, 3)
    
    #####
    gaze_directions = None # 입력 필요 없음. 
    # tracking 된 값들. 
    fit, pred_verts_3d, template_faces = fit_flame(
        verts_3d, # 이것만
        gaze_directions, # 입력 필요 없음. 
        OPENCV2PYTORCH3D.numpy() @ rt, # 학습에 사용 안됨.
        use_jaw_rotation=False, # 그대로
        n_shape_params=150, # 그대로
        n_expr_params=65, # 그대로
        device=args.device, # 그대로
        n_steps=args.step, # 8000, # 그대로
        w_shape_reg=1e-4, # 6 # 그대로
        w_expr_reg=1e-4, # 6 # 그대로
        smooth_eye_rotations=True #is_video, # 그대로
    )

    # flame 2023으로 converting 된 mesh 저장. 
    converted_mesh_dir = output_path / "flowface_mesh"
    converted_mesh_dir.mkdir(exist_ok=True)

    # for frame_id in range(verts_3d.shape[0]):

    #     trimesh.Trimesh(
    #         pred_verts_3d[frame_id], faces=template_faces
    #     ).export(converted_mesh_dir / f"{frame_id:05d}.ply")
    
    def axis_angle_to_matrix_batch(axis_angle):
        """
        axis_angle: (B, 3) numpy array of axis-angle vectors
        returns: (B, 3, 3) rotation matrices
        """
        B = axis_angle.shape[0]

        # theta = magnitude of axis-angle vector
        theta = np.linalg.norm(axis_angle, axis=1)          # (B,)
        small = theta < 1e-8                                # mask for near-zero rotation

        # normalize axis
        axis = np.zeros_like(axis_angle, dtype=np.float32)  # (B, 3)
        axis[~small] = axis_angle[~small] / theta[~small, None]

        x = axis[:, 0]
        y = axis[:, 1]
        z = axis[:, 2]

        c = np.cos(theta)
        s = np.sin(theta)
        C = 1 - c

        # prepare output
        R = np.zeros((B, 3, 3), dtype=np.float32)

        R[:, 0, 0] = c + x*x*C
        R[:, 0, 1] = x*y*C - z*s
        R[:, 0, 2] = x*z*C + y*s

        R[:, 1, 0] = y*x*C + z*s
        R[:, 1, 1] = c + y*y*C
        R[:, 1, 2] = y*z*C - x*s

        R[:, 2, 0] = z*x*C - y*s
        R[:, 2, 1] = z*y*C + x*s
        R[:, 2, 2] = c + z*z*C

        # identity for zero-rotation entries
        R[small] = np.eye(3, dtype=np.float32)

        return R
    def matrix_to_axis_angle_batch(R):
        """
        R: (B, 3, 3) rotation matrices
        returns: (B, 3) axis-angle vectors
        """
        B = R.shape[0]

        # trace-based angle
        trace = np.einsum('bii->b', R)           # (B,)
        cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
        theta = np.arccos(cos_theta)             # (B,)

        # axis from skew-symmetric part
        rx = R[:, 2, 1] - R[:, 1, 2]
        ry = R[:, 0, 2] - R[:, 2, 0]
        rz = R[:, 1, 0] - R[:, 0, 1]

        axis = np.stack([rx, ry, rz], axis=1)    # (B, 3)

        # normalize axis, avoid division by zero
        axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis_norm = np.where(axis_norm < 1e-8, 1.0, axis_norm)
        axis = axis / axis_norm                  # (B, 3)

        # axis-angle = axis * theta
        aa = axis * theta[:, None]               # (B, 3)

        # if theta ~ 0, return zero vector
        aa[np.abs(theta) < 1e-8] = 0.0

        return aa.astype(np.float32)

    flip_front = np.array([
        [1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)

    batch_rot = axis_angle_to_matrix_batch(fit["rot"])
    fliped = matrix_to_axis_angle_batch(flip_front[None, :, :] @ batch_rot)
    
    # convert 파라미터 
    out_flame["rot"] = fliped #np.zeros_like(fit["rot"]) # 초기화 
    out_flame["tra"] = fit["tra"] #np.zeros_like(fit["tra"]) #[ 0.      ,  0.      , -1.469153]
    out_flame["tra"][:, 2] = -1.469153
    out_flame["shape"] = fit["shape"]
    out_flame["expr"] = fit["expr"]
    out_flame["neck_rot"] = fit["neck_rot"]
    out_flame["eye_rot"] = fit["eye_rot"]  # TODO smooth eye rotations

    np.savez(output_path / "fit.npz", **out_flame)

    n_frames = verts_3d.shape[0]

    # If it is a video, that means we can use it for animation. 
    # Save the camera trajectory and create a corresponding camera orbit.
    n_orbit_frames = n_frames
    fps = 25

    print("saving camera trajectories")
    trajectory = {
        "extr": out_flame["extr"].repeat(n_orbit_frames, axis=0), # 가만히 있을려면은 처음꺼 복사?
        "fx": out_flame["fx"].repeat(n_orbit_frames, axis=0),
        "fy": out_flame["fy"].repeat(n_orbit_frames, axis=0),
        "cx": out_flame["cx"].repeat(n_orbit_frames, axis=0),
        "cy": out_flame["cy"].repeat(n_orbit_frames, axis=0),
        "resolution": orig_resolution,
        "fps": fps,
    }
    np.savez(output_path / "cam_static.npz", **trajectory)

    # Create orbit trajectory
    t = np.arange(n_orbit_frames) / fps / ORBIT_PERIOD
    yaw_angles = np.cos(t * 2 * np.pi) * ORBIT_AMPLITUDE_YAW
    pitch_angles = np.sin(t * 2 * np.pi) * ORBIT_AMPLITUDE_PITCH
    for i in range(n_orbit_frames):
        target = out_flame["tra"][0].copy()
        target[1:] = -target[1:]
        trajectory["extr"][i] = pivot_camera_intrinsic(
            trajectory["extr"][i],
            target,
            [yaw_angles[i], pitch_angles[i]]
        )
    np.savez(output_path / "cam_orbit.npz", **trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="path where the converted (FlowFace format) data will be saved",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path where the converted (FlowFace format) data will be saved",
    )
    parser.add_argument(
        "--max_n_ref",
        type=int,
        default=100,
        help="maximum number of reference frames",
    )
    parser.add_argument(
        "--enable_gaze_tracking",
        type=int,
        default=1,
        help="whether to enable gaze tracking (if False, eyeball rotation will be zero)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4000,
        help="device",
    )
    args = parser.parse_args()
    main(args)

# python scripts/pixel3dmm/convert_to_flowface.py --video_path examples/input/felix/images/cam0/ --tracking_path examples/pixel3dmm_tracking/cam0_nV1_noPho_uv2000.0_n1000.0/ --preprocess_path examples/pixel3dmm_tracking/cam0/ --output_path examples/input/felix_converted/ 

# python cap4d/inference/generate_images.py --config_path configs/generation/debug.yaml --reference_data_path examples/input/felix_converted/ --output_path examples/debug_output/felix_converted/

# Create hugging face demo
