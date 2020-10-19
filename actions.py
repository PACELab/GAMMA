import os
import os.path
import logging

SUDO_PASSWORD = 'password'

# max number of COSs in Intel CAT
# check by executing 'pqos -s'
NUM_COS_CAT = 4
NUM_COS_MBA = 8
NUM_NET_CLS = 16

def cpu(id, value, period=100000):
    if value <= 0:
        raise ValueError('Invalid Value!')
    current_script = os.path.realpath(__file__)
    os.system('echo %s | sudo -S python %s' % (SUDO_PASSWORD, current_script))
    
    logging.info('cpu - id: ' + id + ' - delta: ' + value)
    path = '/sys/fs/cgroup/cpu/kubepods/guaranteed/pod' + id + '/cpu.cfs_quota_us'
    f = open(path,"r")
    original = int(f.read())
    curr_value = original + int(value)
    with open(path, "r+") as f:
        data = f.read()
        f.seek(0)
        f.write(curr_value)
        f.truncate()

def memory(id, value, cores):
    if value < 0 or value > NUM_COS_MBA-1:
        raise ValueError('Invalid Value!')
    current_script = os.path.realpath(__file__)
    os.system('echo %s | sudo -S python %s' % (SUDO_PASSWORD, current_script))

    logging.info('memory - id: ' + id + ' - delta: ' + value)
    os.system('sudo ./scripts/association_app ' + value + ' '.join(cores))

def llc(id, value, cores):
    if value < 0 or value > NUM_COS_MBA-1:
        raise ValueError('Invalid Value!')
    current_script = os.path.realpath(__file__)
    os.system('echo %s | sudo -S python %s' % (SUDO_PASSWORD, current_script))

    logging.info('llc - id: ' + id + ' - delta: ' + value)
    os.system('sudo ./scripts/association_app ' + value + ' '.join(cores))

def network(id, value):
    if value < 0 or value > NUM_NET_CLS-1:
        raise ValueError('Invalid Value!')
    current_script = os.path.realpath(__file__)
    os.system('echo %s | sudo -S python %s' % (SUDO_PASSWORD, current_script))

    logging.info('network - id: ' + id + ' - delta: ' + value)
    path = '/sys/fs/cgroup/net_cls/kubepods/guaranteed/pod' + id + '/net_cls.classid'
    # f = open(path,"r")
    # original = int(f.read())
    with open(path, "r+") as f:
        data = f.read()
        f.seek(0)
        f.write('0x001000' + value.zfill(2))
        f.truncate()

def blkio(id, value):
    if value <= 0:
        raise ValueError('Invalid Value!')
    current_script = os.path.realpath(__file__)
    os.system('echo %s | sudo -S python %s' % (SUDO_PASSWORD, current_script))

    # path = '/sys/fs/cgroup/blkio/kubepods/guaranteed/pod' + id + '/blkio.throttle.read_bps_device'
    # path = '/sys/fs/cgroup/blkio/kubepods/guaranteed/pod' + id + '/blkio.throttle.write_bps_device'
    path = '/sys/fs/cgroup/blkio/kubepods/guaranteed/pod' + id + '/blkio.weight'
    with open(path, "r+") as f:
        data = f.read()
        f.seek(0)
        f.write(value)
        f.truncate()
