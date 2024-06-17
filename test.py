from pathlib import Path
import sys

import random
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os 
import PIL as Image

def read_(directory):
    png_images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                png_images.append(img)
    return png_images

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    #images_path = read_png_images('/data/cvpr24-challenge/challenge/data/resources/scene_example/color')
    images_path = []
    directory = '/home/fangj1/Code/go_vocation/data/scene_example/color'
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            images_path.append(image_path)
    #images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512) # images is a list.
    mode = GlobalAlignerMode.PairViewer
    print(images_path)
    if mode == GlobalAlignerMode.PairViewer:
        images_path = images_path[:2]
        print(images_path)

    else: 
        images_path = random.sample(images_path, 10)
        print(images_path)

    images = load_images(images_path, size=640) # images is a list.

    #print(images[0].keys())#dict_keys(['img', 'true_shape', 'idx', 'instance'])
    # print(images[0]['img'].shape)#torch.Size([1, 3, 384, 512])
    # print(images[0]['true_shape'])#[[384 512]]
    # print(images[0]['idx'])
    # print(images[0]['instance'])
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)#list with tuple pairs
    output = inference(pairs, model, device, batch_size=batch_size)
    #print(output.keys())#dict_keys(['view1', 'view2', 'pred1', 'pred2', 'loss'])
    #print(output['view1'].keys())#dict_keys(['img', 'true_shape', 'idx', 'instance'])
    #print(output['pred1'].keys())#dict_keys(['pts3d', 'conf'])
    #print(output['pred1']['pts3d'].shape)#torch.Size([2, 384, 512, 3])
    #print(output['pred1']['conf'])#torch.Size([2, 384, 512])


    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
 

    scene = global_aligner(output, device=device, mode=mode)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    # Retrieve useful values from scene
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # Debug prints to check the contents
    print(f"Images shape: {[img.shape for img in imgs]}")#Images shape: [(480, 640, 3), (480, 640, 3)]
    print(f"Focals: {focals}")
    print(f"Poses: {poses}")
    print(f"3D Points shape: {[pt.shape for pt in pts3d]}")#points shape: [(480, 640, 3), (480, 640, 3)]
    print(f"Confidence masks: {confidence_masks}")

    # Visualize reconstruction with additional checks
    try:
        # Ensure focals and other arrays are not empty
        if focals is not None and len(focals) > 0:
            scene.show()
        else:
            print("Focal lengths are empty or not properly initialized.")
    except IndexError as e:
        print(f"An error occurred: {e}")

    # # find 2D-2D matches between the two images
    # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    # pts2d_list, pts3d_list = [], []
    # for i in range(2):
    #     conf_i = confidence_masks[i].cpu().numpy()
    #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    # print(f'found {num_matches} matches')
    # matches_im1 = pts2d_list[1][reciprocal_in_P2]
    # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # # visualize a few matches
    # import numpy as np
    # from matplotlib import pyplot as pl
    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap('jet')
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # pl.show(block=True)








