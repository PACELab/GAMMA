NUM_ACTIONS = 15
NUM_STATES = 5+2+3

VERBOSE = False

ID = 'default'
THRESHOLD = 0.7
STEP = 10

class AIMD:
    def __init__(self, env):
        self.env = env

    def start(self):
        actions = {}
        actions['cpu_action'] = 0
        actions['mem_action'] = 0
        actions['llc_action'] = 0
        actions['io_action'] = 0
        actions['net_action'] = 0
        while True:
            state, reward, done = self.env.new_step(actions['cpu_action'], actions['mem_action'], actions['llc_action'], actions['io_action'], actions['net_action'], ID)
            curr_arrival_rate = state['curr_arrival_rate']
            cpu_limit = state['cpu_limit']
            mem_limit = state['mem_limit']
            llc_limit = state['llc_limit']
            io_limit = state['io_limit']
            net_limit = state['net_limit']
            curr_cpu_util = state['curr_cpu_util']
            curr_mem_util = state['curr_mem_util']
            curr_llc_util = state['curr_llc_util']
            curr_io_util = state['curr_io_util']
            curr_net_util = state['curr_net_util']
            slo_retainment = state['slo_retainment']
            rate_ratio = state['rate_ratio']
            percentages = state['percentages']
            if VERBOSE:
                print("Update - Current SLO Retainment:", slo_retainment)
                print("Update - Current Util:", str(curr_cpu_util)+'/'+str(cpu_limit), str(curr_mem_util)+'/'+str(mem_limit), str(curr_llc_util)+'/'+str(llc_limit), str(curr_io_util)+'/'+str(io_limit), str(curr_net_util)+'/'+str(net_limit))
            actions = self.getMaxAction(state)

    def getMaxAction(self, state):
        actions = {}
        if state['slo_retainment'] < 1:
            actions['cpu_action'] = state['cpu_limit']
            actions['mem_action'] = state['memory_limit']
            actions['llc_action'] = state['llc_action']
            actions['io_action'] = state['io_action']
            actions['net_action'] = state['net_action']
            return actions
        else:
            actions['cpu_action'] = -STEP
            #actions['cpu_action'] = -state['cpu_limit']*0.05
            actions['mem_action'] = -STEP
            #actions['mem_action'] = -state['memory_limit']*0.05
            actions['llc_action'] = -STEP
            #actions['llc_action'] = -state['llc_action']*0.05
            actions['io_action'] = -STEP
            #actions['io_action'] = -state['io_action']*0.05
            actions['net_action'] = -STEP
            #actions['net_action'] = -state['net_action']*0.05
            return actions
