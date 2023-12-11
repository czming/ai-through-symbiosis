from multiprocessing import Process, Manager
import time
from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 48293


def process_image(img, counter, result):
    time.sleep(0.40)
    result[counter] = img.shape


if __name__ == "__main__":
    with MLSocket() as socket:
        with Manager() as manager:
            counter = 0

            socket.bind((HOST, PORT))
            socket.listen()
            conn, addr = socket.accept()
            result = manager.dict({})
            processes = {}
            t = time.time()
            while True:
                t1 = time.time()
                image = conn.recv(1024)
                if (image == -1).all():
                    break
                p = Process(target=process_image, args=(image, counter, result))
                processes[counter] = p
                counter += 1
                p.start()
            for i in range(counter):
                processes[i].join()
                print(i, result[i])
