import cv2
import imutils.video

import threading
import queue

import numpy as np
import json

import multiprocessing as mp

from flask import Flask, Response, request

BATCH_SIZE = 20 # How many frames are processed at once

app = Flask(__name__)

# Define a function to convert an RGB color to hex format
def process_pixel(pixel):
    if np.isscalar(pixel):
        # Handle scalar input values
        return '%02x%02x%02x' % (pixel, pixel, pixel)
    else:
        # Handle RGB color tuples
        r, g, b = pixel
        return '%02x%02x%02x' % (r, g, b)


# Define a route to get the pixel colors of a video
@app.route('/get_pixels')
def get_pixels():

    gray_scale = str(request.args.get('gray_scale', False)) == 'true'
    video = str(request.args.get('file', 'test.mp4'))

    WIDTH = int(request.args.get('width', 100))
    HEIGHT = int(request.args.get('height', 100))

    print(gray_scale)
    print(video)

    print(WIDTH)
    print(HEIGHT)

    fvs = imutils.video.FileVideoStream(video).start()

    def retrieve_pixels(frame, frames):
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))            

        pixels = []

        if gray_scale:
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            pixels = gray_frame.flatten().tolist()
        else:
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            pixels = np.reshape(rgb_frame, (HEIGHT * WIDTH, 3))

            hex_colors = np.apply_along_axis(process_pixel, 1, pixels)

            pixels = hex_colors.tolist()

        frames.append(pixels)

    def read_frames(fvs, frame_queue):
        while fvs.more():
            frame = fvs.read()
            frame_queue.put(frame)

    frame_queue = queue.Queue()

    t = threading.Thread(target=read_frames, args=(fvs, frame_queue))
    t.daemon = True
    t.start()

    frames = []

    while True:
        frame = frame_queue.get()

        if frame is None:
            # print('stop')
            break

        t = threading.Thread(target=retrieve_pixels, args=(frame, frames))
        t.daemon = True
        t.start()

        # print('next frame')

    t.join()

    cv2.destroyAllWindows()
    fvs.stop()

    frames_data = json.dumps(frames)

    return Response(frames_data, mimetype='application/json')


if __name__ == '__main__':
    app.run()