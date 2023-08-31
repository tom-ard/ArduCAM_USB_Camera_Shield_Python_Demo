import argparse
import time
import signal
import cv2

from Arducam import *
from ImageConvert import *
from arducam_rgbir_debayer import *

exit_ = False

fill_count_dict = {
    0x01: 1,
    0x03: 2,
    0x07: 3,
    0x0F: 4
}

def sigint_handler(signum, frame):
    global exit_
    exit_ = True


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


def display_fps(index):
    display_fps.frame_count += 1

    current = time.time()
    if current - display_fps.start >= 1:
        print("fps: {}".format(display_fps.frame_count))
        display_fps.frame_count = 0
        display_fps.start = current


display_fps.start = time.time()
display_fps.frame_count = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, required=True, help='Specifies the configuration file.')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help='Output device information.')
    parser.add_argument('--preview-width', type=int, required=False, default=-1, help='Set the display width')
    parser.add_argument('-n', '--nopreview', action='store_true', required=False, help='Disable preview windows.')
    

    args = parser.parse_args()
    config_file = args.config_file
    verbose = args.verbose
    preview_width = args.preview_width
    no_preview = args.nopreview

    camera = ArducamCamera()

    if not camera.openCamera(config_file):
        raise RuntimeError("Failed to open camera.")

    if verbose:
        camera.dumpDeviceInfo()

    camera.start()
    camera.setCtrl("setFramerate", 15)
    camera.setCtrl("setExposureTime", 200000)
    # camera.setCtrl("setAnalogueGain", 800)

    scale_width = preview_width

    while not exit_:
        ret, data, cfg = camera.read()

        display_fps(0)

        if no_preview:
            continue

        if ret:
            if camera.cameraCfg["emImageFmtMode"] == 9:
                width = cfg["u32Width"]
                height = cfg["u32Height"]
                bitWidth = cfg["u8PixelBits"]
                if (bitWidth > 8):
                    origin = np.frombuffer(data, dtype=np.uint16)
                else:
                    origin = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)
                rows_fill_count = fill_count_dict.get((camera.color_mode >> 4) & 0x0F, 0)
                cols_fill_count = fill_count_dict.get(camera.color_mode & 0x0F, 0)
                tmp = np.array(fill(origin, height, width, rows_fill_count, cols_fill_count), dtype=np.uint16)
                results = processRgbIr16BitData(tmp, height+4, width+4) 
                rgb_img = RGBToMat(results['rgb'], bitWidth, width+4, height+4)
                ir_full_img = cv2.cvtColor(IRToMat(results['ir_full'], bitWidth, width+4, height+4), cv2.COLOR_GRAY2BGR)
                image = cv2.hconcat([rgb_img[:height, :width], ir_full_img[:height, :width]])
            else: 
                image = convert_image(data, cfg, camera.color_mode)

            if scale_width != -1:
                scale = scale_width / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale)

            cv2.imshow("Arducam", image)
        else:
            print("timeout")

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True
        elif key == ord('s'):
            np.array(data, dtype=np.uint8).tofile("image.raw")

    camera.stop()
    camera.closeCamera()
