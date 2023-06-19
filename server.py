import cv2
import numpy as np
import json

import multiprocessing as mp

import warnings

from sklearn.cluster import MiniBatchKMeans
from flask import Flask, Response, request

BATCH_SIZE = 20 # How many frames are processed at once
COLOR_ACCURACY = 50 # How many colors are allowed for 1 pixel

app = Flask(__name__)

warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 3 to 'auto' in 1.4.")

# Define a function to convert an RGB color to hex format
def process_pixel(pixel):
    if np.isscalar(pixel):
        # Handle scalar input values
        return '%02x%02x%02x' % (pixel, pixel, pixel)
    else:
        # Handle RGB color tuples
        r, g, b = pixel
        return '%02x%02x%02x' % (r, g, b)


def quantize_colors(image, num_colors):
    # Convert the image from the RGB color space to the L*a*b* color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Reshape the image into a feature vector
    h, w = image.shape[:2]
    image = image.reshape((h * w, 3))
    
    # Apply k-means using the specified number of clusters
    clt = MiniBatchKMeans(n_clusters=num_colors)
    labels = clt.fit_predict(image)
    
    # Create the quantized image based on the predictions
    quantized = clt.cluster_centers_.astype("uint8")[labels]
    
    # Reshape the feature vectors to images
    quantized = quantized.reshape((h, w, 3))
    
    # Convert from L*a*b* to RGB
    quantized = cv2.cvtColor(quantized, cv2.COLOR_Lab2RGB)
    
    return quantized


def retrieve_pixels(frame, gray_scale, HEIGHT, WIDTH):
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    if gray_scale:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        pixels = gray_frame.flatten().tolist()
    else:
        quantized_frame = quantize_colors(resized_frame, COLOR_ACCURACY)
        pixels = [process_pixel(pixel) for pixel in quantized_frame.reshape(-1, 3)]
    
    return pixels

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

    cap = cv2.VideoCapture(video)

    frames = []

    with mp.Pool() as pool:
        futures = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            future = pool.apply_async(retrieve_pixels, args=(frame, gray_scale, HEIGHT, WIDTH))
            futures.append(future)

            if len(futures) == BATCH_SIZE:
                for future in futures:
                    frames.append(future.get())

                futures = []

    frames_data = json.dumps(frames)

    cap.release()

    return Response(frames_data, mimetype='application/json')


if __name__ == '__main__':
    app.run()
