from threading import Thread
import time


def thread_j():
    j = 0
    while True:
        j += 1
        print(f'j = {j}')
        time.sleep(1)


if __name__ == '__main__':
    th = Thread(target=thread_j)
    th.daemon = True
    th.start()

    i = 0
    while True:
        i += 1
        print(f'i = {i}')
        time.sleep(1)