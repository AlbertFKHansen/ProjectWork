"""
See connected cameras:
gphoto2 --auto-detect
_____________________
Connect to camera:
gphoto2 --summary

If you get an error when connecting to camera (Error -53), try killing all gvfs daemons with:
pkill gvfsd
Restart the camera and try again.
"""

import subprocess
import os


def capture_image(filename="captured.jpg"):
    # Take a photo and download it
    result = subprocess.run(
        [
            "gphoto2",
            "--capture-image-and-download",
            "--filename", filename,
            "--force-overwrite"
        ],
        stdout=subprocess.DEVNULL
    )


if __name__ == '__main__':
    capture_image()
