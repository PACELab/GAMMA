import sys
import time
# https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        print(item)
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
# https://stackoverflow.com/questions/3160699/python-progress-bar
def show(j, count, prefix="", size=60,  out=sys.stdout):
    if j <= count:
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    else:
        print("\n", flush=True, file=out)
#gen = progressbar(range(15), "Computing: ", 40)
# print("called")
i = 0
while i < 17:
     show(i, 100, "computing")
     i+= 2
     time.sleep(0.1) # any code you need
# i=0
# while i < 15:
#     next(gen)
#     i += 1
#     time.sleep(0.1) # any code you need
