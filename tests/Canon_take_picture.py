"""
Install gphoto2:
sudo apt install gphoto2
________________________
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
import gphoto2 as gp

def capture_image1(filename="captured.jpg"):
    # Take a photo and download it
    subprocess.run(["gphoto2", "--capture-image-and-download", "--filename", filename])


def capture_image2(target_folder='captures'):
    # Ensure output dir exists
    os.makedirs(target_folder, exist_ok=True)

    # Initialize camera
    camera = gp.Camera()
    camera.init()
    print(camera)
    try:
        # Capture image
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
        print(f"Captured to {file_path.folder}/{file_path.name}")

        # Download to local disk
        local_path = os.path.join(target_folder, file_path.name)
        camera_file = camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
        )
        camera_file.save(local_path)
        print(f"Downloaded to {local_path}")

    finally:
        camera.exit()

if __name__ == '__main__':
    print('Press 1 to use function without library\n'
          'Press 2 to use function with library')
    func = input('> ')
    if func == '1':
        capture_image1()
    elif func == '2':
        capture_image2()
