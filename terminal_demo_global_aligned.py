import argparse
import math
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.spatial.transform import Rotation

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import clip
import time
import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--query", type=str, default='chair')###################

    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    # print(len(pts3d)) 
    # print(len(mask))
    # print(len(imgs)) 
    # print(len(cams2world)) 
    # print(len(focals))
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile, scene

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, semantic_segmentations = None):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    if semantic_segmentations is not None:##################        
        #print(rgbimg[0].shape)#        (240, 320, 3)
        #print(type(rgbimg))#<class 'list'>
        #print(type(rgbimg[0]))#<class 'numpy.ndarray'>
        rgbimg = convert_semantics_to_rgb(scene.imgs, semantic_segmentations)
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
def convert_semantics_to_rgb(imgs, semantic_segmentations, mask_threshold=0.8):
    """
    By modifying with Convert semantic segmentations to RGB images
    """
    rgb_imgs = []
    cmap = pl.get_cmap('coolwarm')
    for img, semantic_segmentation in zip(imgs, semantic_segmentations):
        # 假设 img: numpy.ndarray (240, 320, 3)
        # semantic_segmentation.shape: torch.Size([240, 320])
        
        # 将 semantic_segmentation 移动到 CPU
        semantic_segmentation_cpu = semantic_segmentation.cpu().numpy()
        
        # 创建掩膜
        mask = semantic_segmentation_cpu > mask_threshold

        # 生成颜色映射
        colors = cmap(semantic_segmentation_cpu)

        # 将颜色转换为 (240, 320, 3) 的形状
        colors = colors[:, :, :3]  # 保留 RGB 通道，形状为 (240, 320, 3)

        # 创建一个新的图像来保存结果
        result_img = img.copy()

        # 使用掩膜将颜色应用到原图像
        result_img[mask] = colors[mask]
        
        rgb_imgs.append(result_img)

    return rgb_imgs
def load_pixelwise_features(input_img_file):
    """
    Load pixelwise features from a file
    """
    tensors = []
    for path in input_img_file:
        new_path = path.replace('/color/', '/features/').replace('.jpg', '.pt')
        tensor = torch.load(new_path)
        tensors.append(tensor)
    return torch.stack(tensors)
def semantic_segmentation(features, query):
    """
    Perform semantic segmentation on the input image
    return [B,H,W]
    """
    # Load the model
    print(query)
    B, C, H, W = features.shape
    device = features[0].device 
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode text using CLIP text encoder
    text_tokens = clip.tokenize(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).to(torch.float16)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize

    # Prepare an empty classification result image for each batch
    semantic_segmentations = []

    for b in range(B):
        feat_single = features[b]  # Shape [512, H, W]

        # Reshape the feature map
        feat_reshaped = feat_single.view(C, -1).T  # Shape [H*W, 512]

        # Normalize pixel features
        feat_reshaped = feat_reshaped / feat_reshaped.norm(dim=-1, keepdim=True)

        # Compute similarities and measure time
        start_time = time.time()
        with torch.no_grad():
            similarities = torch.matmul(feat_reshaped, text_features.T)  # Shape [H*W, len(text)]
        end_time = time.time()
        print(f"Inference time for batch {b}: {end_time - start_time} seconds")#(240, 320) half of the original image size

        # Find the class with the highest similarity
        print(similarities.shape)
        semantic_segmentation = similarities[:,0].view(H, W)
        semantic_segmentations.append(semantic_segmentation)
    
    return torch.stack(semantic_segmentations)


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, query):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)#[dict_keys(['img', 'true_shape', 'idx', 'instance'])] dict_keys in a list
    #print(imgs[0]['img'].shape)#torch.Size([1, 3, 240, 320])
    #print(imgs[0]['true_shape'].shape)#(1, 2)
    if query:
        features = load_pixelwise_features(filelist)#[512, H,W]
        semantic_segmentations = semantic_segmentation(features,query)#torch.Size([2, 240, 320])
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True) # tuple pairs in a list
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01
    niter = 300

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile, trimesh_scene = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                     clean_depth, transparent_cams, cam_size)
    
    # Display the scene using trimesh
    #trimesh_scene.show()
###########################semamtic segmentation
    if query:
        outfile, trimesh_scene = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                                     clean_depth, transparent_cams, cam_size, semantic_segmentations)
    # Display the scene using trimesh
    trimesh_scene.show()


    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs

def sample_images_uniformly(images, num_samples=15):
    if len(images) < num_samples:
        raise ValueError(f"Number of images ({len(images)}) is less than the number of samples ({num_samples})")
    images = sorted(images)
    interval = len(images) / num_samples
    sampled_images = [images[int(i * interval)] for i in range(num_samples)]
    
    return sampled_images

def main_demo(tmpdirname, model, device, image_size, input_folder, silent=False, query=None):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    
    # List all files in the input folder
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Filter only image files (assuming common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    input_files = [f for f in input_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Check if any image files were found
    if not input_files:
        print("No image files found in the specified folder.")
        return
    input_files = sample_images_uniformly(input_files)#[:2]###########for pair viewer
    #input_files = ['/home/fangj1/Code/go_vocation/data/scene_example/color/1539.jpg', '/home/fangj1/Code/go_vocation/data/scene_example/color/1555.jpg']
    print(input_files)
    # Reconstruction options (same as before)
    schedule = 'linear'
    niter = 'niter'
    scenegraph_type = 'complete'
    winsize = 1 
    refid = 1 
    min_conf_thr = 0.3
    cam_size = 0.05
    as_pointcloud = 1
    mask_sky = 0
    clean_depth = 0
    transparent_cams = 1

    scene, outfile, imgs = recon_fun(input_files, schedule, niter, min_conf_thr, as_pointcloud,
                                     mask_sky, clean_depth, transparent_cams, cam_size,
                                     scenegraph_type, winsize, refid, query)#query added
    print(f"3D model saved to: {outfile}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # Hardcoded input folder
    input_folder = '/data/scannet_2d/scene0012_01/color'#'/home/fangj1/Code/go_vocation/data/scene_example/color'#'/data/cvpr24-challenge/challenge/data/ChallengeDevelopmentSet_converted/42445935/color'#'/data/go_vocation/data/bottle'#
    query = [args.query,'other']
    query = None
    print(query)
    args.image_size = 4096/10#640/2 #original image size is 640*480

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        
        # Run the main demo with input folder specified
        main_demo(tmpdirname, model, args.device, args.image_size, input_folder, silent=args.silent, query=query)