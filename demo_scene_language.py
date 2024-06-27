import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image

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

###################
import sys
import importlib
# Add the root of your project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lang-seg')))

# Now import the module dynamically
module_name = 'lang-seg.extract_lseg_features'
module = importlib.import_module(module_name)

# Access the function
extract_save_lseg_features = getattr(module, 'extract_save_lseg_features', None)
#####################

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
    parser.add_argument("--image_devide", type=int, default=2, help="resize the image to a smaller size")

    return parser


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
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
    return outfile #, scene




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


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, query):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    # modify the variable
    with Image.open(filelist[0]) as img:
        image_size = max(img.size)
    imgs = load_images(filelist, size=image_size/image_devide, verbose=not silent)
    #print(imgs[0]['img'].shape)
    if query:
        query = [query,'other']
        features = load_pixelwise_features(filelist, device)#[512, H,W]
        semantic_segmentations = semantic_segmentation(features, query, device)#torch.Size([2, 240, 320])
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)
    
    # if query:
    #     outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                                  clean_depth, transparent_cams, cam_size, query)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    #print(rgbimg.shape)#list[]
    #print(rgbimg[0].shape)#numpy.ndarray(240, 320, 3)

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

    return scene, outfile, imgs #scene, outmodel, outgallery

def convert_semantics_to_rgb(imgs, semantic_segmentations, mask_threshold=0.8):
    """
    By modifying with Convert semantic segmentations to RGB images
    """
    rgb_imgs = []
    cmap = pl.get_cmap('Greens')#coolwarm
    # print(semantic_segmentations)
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
def load_pixelwise_features(input_img_file,device):
    """
    Load pixelwise features from a file
    """
    tensors = []
    for path in input_img_file:
        #new_path = path.replace('/color/', '/features/').replace('.jpg', '.pt')
        file_name = os.path.basename(path)
        new_file_name = file_name.replace('.jpg', '.pt')
        new_path = os.path.join('temp', 'feat_lseg', new_file_name)        
        tensor = torch.load(new_path).to(device)
        #print(tensor.device)
        tensors.append(tensor)
    return torch.stack(tensors).to(device)
def semantic_segmentation(features, query, device):
    """
    Perform semantic segmentation on the input image
    return [B,H,W]
    """
    # Load the model
    #print(query)
    B, C, H, W = features.shape
    #device = features[0].device 

    # Encode text using CLIP text encoder
    text_tokens = clip.tokenize(query).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).to(torch.float16)
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
        #print(similarities.shape)
        semantic_segmentation = similarities[:,0].view(H, W)
        semantic_segmentations.append(semantic_segmentation)
    
    return torch.stack(semantic_segmentations)

def get_3D_model_with_language_from_scene(outdir, silent, device = 'cuda', scene = None, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, query = None, filelist = None, mask_threshold=0.8):
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

    if query:
        query = [query,'other']
        features = load_pixelwise_features(filelist, device)#[512, H,W]
        semantic_segmentations = semantic_segmentation(features, query, device)#torch.Size([2, 240, 320])

    if semantic_segmentations is not None:##################        
        #print(rgbimg[0].shape)#        (240, 320, 3)
        #print(type(rgbimg))#<class 'list'>
        #print(type(rgbimg[0]))#<class 'numpy.ndarray'>
        rgbimg = convert_semantics_to_rgb(scene.imgs, semantic_segmentations, mask_threshold)
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    model_from_scene_fun_with_query = functools.partial(get_3D_model_with_language_from_scene, tmpdirname, silent, device)
    
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            

            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref"],
                                                  value='complete', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)


            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=False, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
            run_btn = gradio.Button("Reconstruction")
            
            outmodel = gradio.Model3D()

            extract_btn = gradio.Button("Extract features")

            mask_threshold = gradio.Slider(label="mask_threshold", value=0.85, minimum=0.0, maximum=1.0, step=0.05)

            query_input = gradio.Textbox(label="Query", placeholder="Type a query here")

            query_btn = gradio.Button("Segmentation")

            outgallery = gradio.Gallery(label='rgb,depth,confidence', columns=3, height="100%")

            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, refid],
                          outputs=[scene, outmodel, outgallery])
            
            def process_files(files, progress=gradio.Progress(track_tqdm=True)):
                progress(0, desc="Extracting features")
                extract_save_lseg_features(files, progress, devide=2)
                progress(1, desc="Feature extraction completed")
                #return "Feature extraction completed. Extracted features saved at ./temp/feat_lseg"
                return "Feature extraction completed. Extracted features saved at ./temp/feat_lseg"
            
            extract_btn.click(fn=process_files,
                          inputs=[inputfiles], 
                          outputs=[extract_btn],
                          show_progress=True)
            
            query_btn.click(fn=model_from_scene_fun_with_query,
                          inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                clean_depth, transparent_cams, cam_size, query_input, inputfiles, mask_threshold],
                          outputs=[outmodel])
            
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            
            mask_threshold.change(fn=model_from_scene_fun_with_query,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, query_input, inputfiles, mask_threshold],
                            outputs=outmodel)

            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size],
                                    outputs=outmodel)
    demo.launch(share=False, server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    global image_devide
    image_devide = args.image_devide

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
    #load clip model
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)

    # dust3r will write the 3D model inside tmpdirname
    # 获取当前工作目录的路径
    current_working_directory = os.getcwd()

    # 指定临时目录的创建路径为当前目录下的 /tmp 文件夹
    temporary_directory_path = os.path.join(current_working_directory, 'temp')
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo', dir=temporary_directory_path) as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)
