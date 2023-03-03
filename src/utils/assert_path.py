import os
import os.path as osp


def assert_path(dir_path):
    dirs = dir_path.split("/")
    for i in range(len(dirs)):
        cur_path = osp.join(*dirs[:i + 1])
        if not osp.exists(cur_path):
            os.makedirs(cur_path)
