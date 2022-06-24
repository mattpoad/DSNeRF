import numpy as np
import os, imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, imgs_type='images', factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, imgs_type+'_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, imgs_type+'_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, imgs_type)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()


    for r in factors + resolutions:
        if isinstance(r, int):
            name = imgs_type+'_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = imgs_type+'_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

        
        
def _load_data(basedir, directory='images', downsample=True, factor=None, width=None, height=None, load_imgs=True, masksasimage=False, i_masks=None, i_masks_poses=None, recenter=True, test=False):

    img0 = [os.path.join(basedir, directory, f) for f in sorted(os.listdir(os.path.join(basedir, directory))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''

    if downsample:
        if factor is not None and factor>1:
            sfx = '_{}'.format(factor)
            _minify(basedir, imgs_type=directory, factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(sh[1] / factor)
            _minify(basedir, imgs_type=directory, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(sh[0] / factor)
            _minify(basedir, imgs_type=directory, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        else:
            factor = 1
    
    
    imgdir = os.path.join(basedir, directory + sfx)
        
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    if i_masks is not None:
        imgfiles = [imgfiles[i] for i in i_masks]

    if os.path.isfile(os.path.join(basedir, 'poses_bounds.npy')):
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    elif test:
        poses_arr = np.zeros((len(imgfiles), 17))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor


    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # N x 3 x 5

    # if recenter:
    #     poses = recenter_poses(poses)
    
    if test:
        poses[0, :3, 3] = np.array([poses.shape[0]*0.001/2, 1.*poses.shape[0]*0.04/2, 1.*poses.shape[0]*0.16/2])
        poses[0, :3, :3] = np.diag(np.array([1,1,1]))
        for i in range(1,poses.shape[0]):
            poses[i, :3, :3] = np.diag(np.array([1,1,1]))
            poses[i, :3, 3] = poses[i-1, :3, 3] + np.array([-0.001, -0.04, -0.16])
    
    if directory == 'masks' and i_masks is not None: #
        if i_masks_poses == None:
            poses = np.array([poses[i, :, :] for i in i_masks])
        else:
            poses = np.array([poses[i, :, :] for i in i_masks_poses])

            
    if poses.shape[0] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[0]) )
        return

    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    if directory=='masks':
        imgs = imgs = [imread(f) for f in imgfiles]
    else:
        imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]

    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs



def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


def poses_linear(poses):
    hwf = poses[0, :3, -1:]
    
    print(f"final: poses[:, :3, 3]max {poses[:, :3, 3][:, 2].argmax()}")
    print(f"init: poses[:, :3, 3]min {poses[:, :3, 3][:, 2].argmin()}")

    init_idx = poses[:, :3, 3][:, 2].argmin()
    final_idx = poses[:, :3, 3][:, 2].argmax()

    init_pos = poses[:, :3, 3][init_idx]
    final_pos = poses[:, :3, 3][final_idx]
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    
    return hwf, vec2, up, init_pos, final_pos

def poses_translation(poses):

    pos = poses[:, :3, 3]
    T = np.zeros((len(poses)-1, 3))

    for i in range(1, len(poses)):
        T[i-1] = pos[i] - pos[i-1]

    return normalize(T.sum(0))


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)  # camera position eye
        # np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])) center
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))  # direction
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1)) 
    return render_poses


def render_path_linear(poses, N, focal=0, sideview=False, freq_sv=30, N_sv=20):
    render_poses = []
    
    hwf, vec2, up, init_pos, final_pos = poses_linear(poses)

    if sideview:
        T = poses_translation(poses)
        print(f"---------T {T}")
        print(f"---------T shape {T.shape}")
        print(f"up shape {up.shape}")
        tt = poses[:,:3,3]
        rads = np.percentile(np.abs(tt), 90, 0)
    
    for i,x in enumerate(np.linspace(0., 1., N)):
        new_pos = x*(final_pos - init_pos) + init_pos
        new_c2w = np.concatenate([viewmatrix(vec2, up, new_pos), hwf], 1)
        render_poses.append(new_c2w)
        
    if sideview: # and i%freq_sv==0:
        #render_poses += render_path_sideview(new_c2w, up, rads, focal, T, N_sv)
        render_poses_test = render_path_test(poses, N)[::-1]
        r0 = render_poses_test[-1]
        deltac2w = render_poses[0] - r0
        for i in np.linspace(0, 1, 10)[1:-1]:
            newc2w = i*deltac2w + render_poses[0]
            
    return render_poses


def render_path_test(poses, N, focal=0):

    print(f"poses[:, :3, 0] {poses[:, :3, 0]}")

    render_poses = []
    hwf, vec2, up, init_pos, final_pos = poses_linear(poses)
    T = poses_translation(poses)
    print(f"T {T}")
    T_ = np.copy(T)
    T_[0] = 0
    T_[1] = 0     # view adjustment along the vertical dimension
    T_[2] = 1     # view adjustment along the horizontal dimension
    #T_[0] *= 0
    print(f"T_ {T_}")

    print(f"norm vec2 {normalize(vec2)}")
    print(f"up {up}")

    pos = init_pos + T_
    #for x in np.linspace(0., 1., N+1)[:-1]:
    for theta in np.linspace(0., 2. * np.pi, N):
        x = theta/(2. * np.pi)
        new_pos = x*(final_pos - init_pos) + init_pos
        print(f"newpos {new_pos - pos}")
        pos = new_pos
        new_c2w = np.concatenate([viewmatrix(vec2, up, new_pos), hwf], 1)
        c = np.dot(new_c2w[:3,:4], np.array([np.cos(2*np.pi), -np.sin(2*np.pi), 0, 1]))  # 2pi : droite   pi : gauche
        z = normalize(c - np.dot(new_c2w[:3,:4], np.array([0,0,-focal, 1.])))
        #render_poses.append(new_c2w)
        #print(f"c {c}")
        #print(f"z {z}")
        #c = np.dot(new_c2w[:3,:4], np.array([np.cos(1*np.pi), -np.sin(1*np.pi), 0, 1]))  # 2pi : droite   pi : gauche
        #z = normalize(c - np.dot(new_c2w[:3,:4], np.array([0,0,-focal, 1.])))
        #z[2] *= 0.7
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1)) 
        
    return render_poses



def render_path_sideview(c2w, up, rads, focal, T, N):

    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    pos = c2w[:3, 3]
    c2w = viewmatrix(T, up, pos)
    up = c2w[:3, 1]

    for theta in np.linspace(0., 2. * np.pi, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta), 1.]) * rads)  # camera position eye
        # np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]) center
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))  # direction
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1)) 
    
    return render_poses


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses



def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds



def load_llff_data(imgs_type, basedir, downsample=True, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, linear=False, sideview=False, i_masks=None, i_masks_poses=None, test=False, test_traj=False) :
    
    if imgs_type == 'images':
        poses, bds, imgs = _load_data(basedir, downsample=downsample, factor=factor, test=test) # factor=8 downsamples original imgs by 8x
    elif imgs_type == 'masks':
        print(f"i_masks {i_masks}")
        print(f"i_masks_poses {i_masks_poses}")
        poses, bds, imgs = _load_data(basedir, directory='masks', downsample=downsample, factor=factor, i_masks=i_masks, i_masks_poses=i_masks_poses, test=test)

    elif imgs_type == 'masksasimages':
        poses, bds, imgs = _load_data(basedir, directory='images', factor=factor, masksasimage=True)
        
    print('Loaded', basedir, bds.min(), bds.max())
    
    # print('poses_bound.npy:\n', poses[:,:,0])

    # Correct rotation matrix ordering and move variable dim to axis 0
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    print(poses.shape)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print("bds:", bds[0])
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    # print('before recenter:\n', poses[0])

    if recenter and not test:
        poses = recenter_poses(poses)

    if linear:
        if test_traj:
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focal = mean_dz

            N_view = 100
            render_poses = render_path_test(poses, N_view, focal=focal)

        else:
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focal = mean_dz
            print(f"-------focal {focal}")

            N_views = 100
            
            render_poses = render_path_linear(poses, N=N_views, focal=focal, sideview=sideview)
            #render_poses = render_path_linear(poses, N=N_views, sideview=sideview)
        
            
        
    elif spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz
        print(f"-------focal {focal}")
        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2
            N_views = int(N_views)

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    if imgs_type=='masks':
        images = images.astype(np.int)
    else:
        images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

def load_sensor_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depths = [imageio.imread(f) for f in depthfiles]
    depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

    

    

