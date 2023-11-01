import os
import subprocess
import traceback


def extract_tar_file(tar_file, dest, create_subdir=True):
    """
    Extract a tar_file at the destination location represented by 'dest'.
    If dest is None, use the cwd as dest.
    if create_subdir is True, create a subdirectory inside dest in which the tar file will be extracted.
    Returns the dir path in which the tar file was extracted.
    """

    try:
        tarfname = os.path.basename(tar_file).split(".tar")[0]

        # Set the destination directory based on input args
        if dest is not None and create_subdir is True:
            dest = os.path.join(os.path.abspath(dest), tarfname)
        if dest is None:
            dest = os.path.join(os.getcwd(), tarfname)

        # Create the dest directory
        os.makedirs(dest, exist_ok=True)

        # Launch the tar command to extract the tar file
        untar_cmd = "tar -xf {} -C {}".format(tar_file, dest)
        p = subprocess.run(untar_cmd.split())
        p.check_returncode()

        return dest

    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        raise e
