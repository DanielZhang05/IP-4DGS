import sys
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, ModelHiddenParams, PipelineParams
from argparse import ArgumentParser
from utils.grouping import group_gaussians_by_semantics, export_groups_to_ply, compute_group_centers, project_points
from utils.params_utils import merge_hparams
import torch
import matplotlib
from gaussian_renderer import render
import numpy as np
import cv2

parser = ArgumentParser()
lp = ModelParams(parser)
hp = ModelHiddenParams(parser)
pp = PipelineParams(parser)

parser.add_argument('--load_stage', type=str, default='fine-lang-discrete')
parser.set_defaults(model_path="output/hypernerf/americano/americano_0")
parser.set_defaults(source_path="data/hypernerf/americano")
parser.add_argument('--configs', type=str, default='arguments/hypernerf/default.py')
parser.add_argument('--language_feature_hiddendim', type=int, default=6)
parser.add_argument('--export_2d', type=bool, default=True)
parser.add_argument('--vis2d_source', type=str, default='language')
parser.add_argument('--render_grouping', type=bool, default=True)
parser.add_argument('--target_gid', type=int, default=4)
args = parser.parse_args()

if args.configs:
    import mmcv
    config = mmcv.Config.fromfile(args.configs)
    args = merge_hparams(args, config)

model_params = lp.extract(args)
hyper = hp.extract(args)
pipeline = pp.extract(args)

import os as _os
_os.environ["language_feature_hiddendim"] = str(args.language_feature_hiddendim)

hyper.net_width = 128
hyper.defor_depth = 1

# load module
print("Start Loading Model")
gaussians = GaussianModel(sh_degree=3, args=hyper)
scene = Scene(model_params, gaussians, load_iteration=-1, load_stage=args.load_stage)
print("Model Loaded Successfully")


# grouping
print("Start Getting Features")
pc = scene.gaussians
try:
    language_features = pc.get_language_feature
except Exception as e:
    print("Failed to get language features. Ensure the loaded stage includes 'lang' and a trained checkpoint exists.")
    print(str(e))
    sys.exit(1)
xyz = pc.get_xyz
opacity = pc.get_opacity
print("Features Got Successfully")


print("Start Grouping")
method = 'similarity'
groups = group_gaussians_by_semantics(
    language_features,
    method=method,
    n_clusters=10,
    similarity_threshold=0.5,
    debug=False,
)
print("Grouping Done Successfully")
if len(groups) > 0:
    print(f"  - Number of Gaussians: {len(groups)}")
    print(f"  - Number of Groups: {groups.max().item() + 1}")
else:
    print("  - No grouping results")

# export viusal result ply
import os
out_dir = os.path.join(model_params.model_path, "point_cloud")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"grouping_{args.load_stage}_{method}.ply")
export_groups_to_ply(xyz, groups, out_path)
print(f"Saved grouping visualization to: {out_path}")

# compute centers
centers = compute_group_centers(xyz, opacity, groups)
print("Centers Computed Successfully")
print(f"  - Centers count: {centers.shape[0]}")
for i in range(min(5, centers.shape[0])):
    c = centers[i]
    print(f"    center[{i}]: [{c[0].item():.4f}, {c[1].item():.4f}, {c[2].item():.4f}]")

def render_grouping_visualization(
    scene,
    gaussians,
    pipeline,
    model_params,
    args,
    groups,
    centers,
    out_dir,
    method,
    target_gid=None
):
    print("Rendering grouping result...")
    # Get camera
    if len(scene.getTrainCameras()) > 0:
        view = scene.getTrainCameras()[0]
    elif len(scene.getTestCameras()) > 0:
        view = scene.getTestCameras()[0]
    else:
        print("No cameras found for rendering.")
        sys.exit(1)
    
    # Generate colors
    cmap = matplotlib.colormaps["tab20"]
    N = groups.shape[0]
    colors = torch.zeros((N, 3), device="cuda", dtype=torch.float32)
    
    # filter valid groups
    valid_mask = groups >= 0
    valid_groups = groups[valid_mask]
    
    # Map group IDs to RGB
    num_groups = int(groups.max().item()) + 1
    palette = torch.zeros((num_groups, 3), device="cuda", dtype=torch.float32)
    
    for i in range(num_groups):
        rgba = cmap(i % 20)
        palette[i] = torch.tensor(rgba[:3], device="cuda")
        
    colors[valid_mask] = palette[valid_groups]
    if target_gid is not None:
        mask = (groups == target_gid)
        colors[~mask] = torch.tensor([0.7, 0.7, 0.7], device=colors.device, dtype=colors.dtype)

    # Render
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    rendering = render(view, gaussians, pipeline, background, None, override_color=colors, cam_type=scene.dataset_type, args=args, stage=args.load_stage)["render"]

    # Project all centers using project_points, we'll draw only target_gid
    points2d, valid_mask = project_points(
        centers,
        view.full_proj_transform,
        int(view.image_width),
        int(view.image_height)
    )

    # Draw circles on rendering

    # Convert to HWC numpy uint8
    img_np = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

    # Draw circles
    points2d_np = points2d.detach().cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()

    H, W = int(view.image_height), int(view.image_width)

    for i in range(points2d_np.shape[0]):
        if not valid_mask_np[i]:
            continue
        if target_gid is not None and i != target_gid:
            continue
        x, y = int(points2d_np[i, 0]), int(points2d_np[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img_np, (x, y), 4, (0, 0, 255), -1)

    # Save
    out_img_path = os.path.join(out_dir, f"render_grouping_{args.load_stage}_{method}.png")
    cv2.imwrite(out_img_path, img_np)
    print(f"Saved rendering to {out_img_path}")

# render grouping
if args.render_grouping:
    render_grouping_visualization(
        scene,
        scene.gaussians,
        pipeline,
        model_params,
        args,
        groups,
        centers,
        out_dir,
        method,
        target_gid=args.target_gid
    )
