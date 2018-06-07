import time

totaltime = {}
start_at = {}


def tik(name):
    start_at[name] = int(round(time.time() * 1000))


def tok(name):
    if name not in start_at:
        raise Exception("not tik yet")
    if name not in totaltime:
        totaltime[name] = 0.
    totaltime[name] += int(round(time.time() * 1000)) - start_at[name]


def print_time(name=None):
    print('------- running time -------')
    if name is not None:
        print(name, totaltime[name])
    else:
        for name,t in totaltime.items():
            print('---', name, t)
    print('---------------------------')


def reset():
    global totaltime
    global start_at
    totaltime = {}
    start_at = {}
