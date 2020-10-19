#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sched.h>
#include <set>
#include <vector>

// PRIu32, PRIu64
#include <inttypes.h>

// syscall(SYS_gettid)
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

// cpu_set
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define NS_PER_S (1000000000L)
#define USEC 1000000

uint64_t getNs() {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec*NS_PER_S + ts.tv_nsec;
}

int main(int argc, const char** argv) {
	// Usage: ./cpu <duration in sec> <core1,core2,...>
	if (argc < 2) { 
		printf("Usage: ./cpu <duration in sec> <core1,core2,...>\n"); 
		exit(0);
	}

	uint32_t maxThreads = omp_get_num_procs();
	printf("Total number of processors (cores): %" PRIu32 "\n", maxThreads);

	std::set<int> cores_input;
	for (int i = 0; i < argc - 2; i++) {
		int core = atoi(argv[i+2]);
		if (core >= 1 && core <= maxThreads) {
			cores_input.insert(core);
		}
	}
	std::vector<int> cores(cores_input.begin(), cores_input.end());

	uint64_t nsTotal = NS_PER_S*atoi(argv[1]); // ns
	uint64_t endNs = nsTotal > 0 ? getNs() + nsTotal : 0;

	if (cores.size() == 0) {
		// all cores
		#pragma omp parallel for
		for (int i = 0; i < maxThreads; i++) {
			printf("Core #%d, TID: %ld\n", i, syscall(SYS_gettid));

			// bind to a particular core
			cpu_set_t *cpusetp;
			cpusetp = CPU_ALLOC(maxThreads);
			if (cpusetp == NULL) {
				perror("CPU_ALLOC");
				exit(EXIT_FAILURE);
			}
			CPU_ZERO_S(maxThreads, cpusetp);
			CPU_SET_S(i, maxThreads, cpusetp);
			if (sched_setaffinity(syscall(SYS_gettid), sizeof(*cpusetp), cpusetp) == -1) {
				perror("sched_setaffinity");
				exit(EXIT_FAILURE);
			}
			CPU_FREE(cpusetp);

			// empty while-loop
			if (endNs > 0) {
				while (getNs() < endNs);
			} else {
				// endless mode
				while (1);
			}
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < cores.size(); i++) {
		int core = cores[i];
		printf("Core #%d, TID: %ld\n", core, syscall(SYS_gettid));

		cpu_set_t *cpusetp;
		cpusetp = CPU_ALLOC(maxThreads);
		if (cpusetp == NULL) {
			perror("CPU_ALLOC");
			exit(EXIT_FAILURE);
		}
		CPU_ZERO_S(maxThreads, cpusetp);
		CPU_SET_S(core-1, maxThreads, cpusetp);
		if (sched_setaffinity(syscall(SYS_gettid), sizeof(*cpusetp), cpusetp) == -1) {
			perror("sched_setaffinity");
			exit(EXIT_FAILURE);
		}
		CPU_FREE(cpusetp);

		// empty while-loop
		if (endNs > 0) {
			while (getNs() < endNs);
		} else {
			// endless mode
			while (1) {
				// intensity adjuster
				// usleep(USEC);
			}
		}
	}

	return 0;
}
