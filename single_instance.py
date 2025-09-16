import os, sys, fcntl

class SingleInstance:
    def __init__(self, lockfile=None):
        if lockfile is None:
            # Default: one lock per script name
            script = os.path.basename(sys.argv[0])
            lockfile = f"/tmp/{script}.lock"
        self.lockfile = lockfile
        self.fp = open(self.lockfile, "w")

    def acquire(self):
        try:
            fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            print("--------------------------------------------------------------------------------")
            print(f"Another instance of {self.lockfile} is already running!")
            print("--------------------------------------------------------------------------------")
            sys.exit(1)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Lock auto-released by OS if process exits/crashes
        self.fp.close()
