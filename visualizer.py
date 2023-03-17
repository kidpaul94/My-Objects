import copy
import argparse
import numpy as np
import open3d as o3d

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='visualizer')
    parser.add_argument('--dataset', default='t-less', type=str,
                        help='itodd, others, sileane, or t-less.')
    parser.add_argument('--object', default='4096_obj_02', type=str,
                        help='object file name.')

    global args
    args = parser.parse_args(argv)

def sequential_visualizer(path_dataset: str, obj: str) -> None:
    """
    Squentially visualize a full grasp configuration w.r.t an object.
    
    Parameters
    ----------
    path_dataset : str
        name of a dataset we wish to visualize
    obj : str
        object file name

    Returns
    -------
    None
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    gripper = o3d.io.read_triangle_mesh('./stl/gripper.stl')
    gripper.paint_uniform_color([1, 0, 0])

    path_aprvs = f'./grasp_dict/{path_dataset}/{obj}/{obj}_aprvs.txt' 
    path_cpps = f'./grasp_dict/{path_dataset}/{obj}/{obj}_cpps.txt'
    pcd = o3d.io.read_point_cloud(f'./pcd/{path_dataset}/{obj}.pcd')

    with open(path_cpps) as f1:
        cpps = eval(f1.read())
    cpps = np.asarray(cpps).T

    with open(path_aprvs) as f2:
        aprvs = eval(f2.read())

    centers = 0.5 * (cpps[:3,:] + cpps[3:6,:])
    directions = cpps[:3,:] - cpps[3:6,:]

    idx = len(aprvs)

    for i in range(idx):
        temp = aprvs[i]

        temp_center = centers[:,i]
        temp_direction = directions[:,i] / np.linalg.norm(directions[:,i])

        for aprv in temp:
            configuration = copy.deepcopy(pcd)
            last_ax = np.cross(aprv, temp_direction)
            R = np.eye(3)
            R[:,0], R[:,1], R[:,2] = temp_direction, last_ax, aprv

            res = np.eye(4)
            res[:3,:3] = R.T
            res[:3,3] = -R.T @ temp_center * 0.001

            configuration.transform(res)
            o3d.visualization.draw_geometries([configuration, gripper, frame])
            break

if __name__ == "__main__":
    parse_args()
    sequential_visualizer(path_dataset=args.dataset, obj=args.object)