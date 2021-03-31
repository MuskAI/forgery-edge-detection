import numpy as np
import time
a = np.random.randn(320,320)
a = a.round()
list = []
start = time.time()
for x in range(320):
    for y in range(320):
        if a[x,y] == 1:
            list.append([x,y])

end = time.time()
print(end-start)


start = time.time()
b = a.flatten()
loc = np.where(b==1,print(123))
end = time.time()
print(end-start)
print(loc)

def print():
    print(123)
def print2():
    print(321)