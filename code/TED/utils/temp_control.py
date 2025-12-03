import os
import tempfile
import shutil

class TempDir():
    def __init__(self):
        self.tmp_dir = None

    def __enter__(self, work_dir='.'):
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        self.tmp_dir = tempfile.mkdtemp(dir=work_dir)
        return self.tmp_dir
    
    def __exit__(self):
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir + '/', ignore_errors=True)
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

def split_sdfs(sdfs_string):
    sdf_string_list, sdf_string = [], []

    sdfs_string_lines = sdfs_string.splitlines()
    for l in sdfs_string_lines:
        l_items = l.split()
        sdf_string.append(l)
        if len(l_items) == 1 and l_items[0] == "$$$$":
            sdf_string_list.append("\n".join(sdf_string))
            sdf_string = []
            continue
    return sdf_string_list