import tensorflow as tf
import time
import os
import sys
import numpy as np
import cv2


def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "", iterable_size=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if iterable_size:
        total = iterable_size
    else:
        total = len(iterable)

    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        if percent != ("{0:." + str(decimals) + "f}").format(100 * ((iteration-1) / float(total))):
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            sys.stderr.write(f'\r{prefix} |{bar}| {percent}% {suffix}' + printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    sys.stderr.flush()
    #print('d!')


def prepare_resized_dataset(METU_RAW_PATH, METU_DATASET_PATH, inputFilesList=None, xScale=128, yScale=128, margin=0):
    start_time = time.time()
    imgList = inputFilesList or tf.data.Dataset.list_files(METU_RAW_PATH+"*")
    if not os.path.exists(METU_DATASET_PATH):
        os.mkdir(METU_DATASET_PATH)
    for imgPath in progressBar(imgList, prefix = 'Progress:', suffix = 'Complete', length = 50):
        img = cv2.imread(imgPath)
        x,y,z = img.shape

        grCoord = max(x,y)
        X = int(np.round(x/(grCoord/(xScale-2*margin))))
        Y = int(np.round(y/(grCoord/(yScale-2*margin))))
        whiteImg = np.ones((xScale,yScale,3))*255
        rescImg = cv2.resize(img, dsize=(Y, X), interpolation=cv2.INTER_CUBIC)
        #rescImg = rescale(rescImg)
        middleX = (xScale-X)//2
        middleY = (yScale-Y)//2
        
        whiteImg[middleX:middleX+X, middleY:middleY+Y] = rescImg[:,:]
        save_path = f"{METU_DATASET_PATH}/{os.path.basename(imgPath)}"
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, whiteImg)

    end_time = time.time()

    print(f'Time taken to resize dataset: {end_time-start_time} seconds')


def rescale(nparray, scale=128.0, translate = 128.0):
    return (np.array(nparray, dtype=np.float32)- translate)/scale


def load_sample(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rescale(img)


def load_dataset(metu_dataset_path, portion=1.0, nexamples=None, scale=128.0, translate=128.0):
    start_time = time.time()
    imgList = os.listdir(metu_dataset_path) 
    total_examples = len(imgList)
    to_load = min(nexamples or int(portion*total_examples), total_examples)

    images_array = np.zeros(shape=(to_load, 128, 128, 3), dtype=np.float32)

    for i, imgPath in enumerate(progressBar(imgList[:to_load], prefix = 'Progress:', suffix = 'Complete', length = 50)):
        images_array[i] = load_sample(f'{metu_dataset_path}/{imgPath}')

    end_time = time.time()

    print(f'Time taken to load {to_load} samples from METU dataset: {end_time-start_time:.3} seconds')
    return tf.data.Dataset.from_tensor_slices(images_array)


def save_plot_plt(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])
    if not os.path.exists(SAMPLES_PATH):
        os.mkdir(SAMPLES_PATH)
    filename = f"{SAMPLES_PATH}/generated_plot_epoch-{epoch+1}.png"
    plt.savefig(filename)
    plt.close()


def save_plot(examples, path, epoch, n, size, plotName = None):
    IMG_W, IMG_H, IMG_C = size
    dw = IMG_W+1
    dh = IMG_H+1
    canvas = np.zeros((dw*n, dh*n, IMG_C), dtype=np.float32)
    for i in range(n * n):
        r = i % n
        c = i // n
        canvas[r*dw:r*dw+IMG_W, c*dh:c*dh+IMG_H, :] = examples[i]
    
    canvas = (canvas * 128.0)+128.0
    canvas = canvas.astype(np.uint8)
    if not os.path.exists(path):
        os.mkdir(path)
    if plotName:
        filename = f"{path}/{plotName}-{epoch+1}.png"
        tf.io.write_file(filename, tf.io.encode_png(canvas))
    return canvas


def tf_db_to_array(tfdb, tfdb_size):
    shape = tf.shape(next(iter(tfdb.take(1))))
    batch_size = shape[0]
    my_array = np.empty(tuple([tfdb_size, *shape[1:]]))
    for i, el in enumerate(progressBar(iter(tfdb), iterable_size = (tfdb_size + batch_size-1)//batch_size)): 
        t = i*batch_size
        my_array[t:t+el.shape[0], :] = el
    return my_array