import logging
import threading
import time
import subprocess
import os


# Configure logging to write logs to a file
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Bottlenecks:
    def __init__(self, node, measure_list, duration_list):
        self.node = node
        self.measure_list = measure_list
        self.duration_list = duration_list

def creating_bottlenecks(bottlenecked_nodes, interference_percentage, phases, experiment_folder):
    threads = []
    for i, node in enumerate(bottlenecked_nodes):
        threads.append(threading.Thread(target=create_CPU_bottlenecks, args=(Bottlenecks(node,interference_percentage, phases), experiment_folder)))
        threads[i].start()
    return threads

def create_CPU_bottlenecks(bottleneck, destination):
    """
    duration_list should have periods of non-bottleneck and bottleneck phases.
    """
    # TODO: assert that the sum of phases is greater than the experiment duration and the extra time.
    logging.debug(f"Starting thread {threading.current_thread().name}")
    try:
        with open(os.path.join(destination, f"{bottleneck.node}_phases"), "w") as f:
            for i in range(len(bottleneck.duration_list)):
                if i%2 ==0:
                    logging.debug(f"{threading.current_thread().name} sleeping for {bottleneck.duration_list[i]}")
                    os.system(f"sleep {bottleneck.duration_list[i]}")
                else:
                    logging.debug(f"{threading.current_thread().name} creating bottleneck for {bottleneck.duration_list[i]}")
                    f.write(f"Bottleneck of type CPU with measure {bottleneck.measure_list[i//2]/100} starts at {time.time()}\n")
                    # load percentage should be [0.1]
                    command = f"python3 /home/ubuntu/firm_compass/tools/CPULoadGenerator/CPULoadGenerator.py -l {bottleneck.measure_list[i//2]/100} -d {bottleneck.duration_list[i]} -c 0 -c 1 -c 2 -c 3"
                    p = subprocess.run(f'ssh -i /home/ubuntu/compass.key ubuntu@{bottleneck.node} "{command}"', stdout=subprocess.PIPE, shell=True, check=True)
                    f.write(f"Bottleneck of type CPU with measure {bottleneck.measure_list[i//2]/100} ends at {time.time()}\n")
    except Exception as e:
        logging.exception(f"{threading.current_thread().name} Exception occurred in thread for {bottleneck.node}: {e}")

logging.info("Creating bottlenecks")
threads = creating_bottlenecks(["userv4", "userv7"], [60, 99, 50], [240, 300, 120, 120, 360, 240], "/mnt/experiments/realistic_july31_25min_repeat_800_1")

print(threading)
for thread in threads:
    print("is thread alive :",thread.is_alive())
    thread.join()