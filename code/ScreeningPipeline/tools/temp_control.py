import tempfile
import shutil
import os

class TempDir():
    def __init__(self):
        self.tmp_dir = None

    def __enter__(self, work_dir):
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        self.tmp_dir = tempfile.mkdtemp(dir=work_dir)
        return self.tmp_dir
    
    def __exit__(self):
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir)
