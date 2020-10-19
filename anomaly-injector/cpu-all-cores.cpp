#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>

#define NS_PER_S (1000000000L)

uint64_t getNs() {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec*NS_PER_S + ts.tv_nsec;
}

int main(int argc, const char** argv) {
	// Usage: ./cpu-all-cores <duration in sec>
	if (argc < 2) { 
		printf("Usage: ./cpu-all-cores <duration in sec>\n"); 
		exit(0);
	}
	
	uint32_t maxThreads = omp_get_num_procs();
	printf("Total number of processors (cores): %" PRIu32 "\n", maxThreads);

	uint64_t nsTotal = NS_PER_S*atoi(argv[1]);

	printf("Running with %d threads for %d seconds\n", maxThreads, atoi(argv[1]));
	omp_set_num_threads(maxThreads);
	uint64_t endNs = nsTotal > 0 ? getNs() + nsTotal : 0;
        
        #pragma omp parallel
	if (endNs > 0) {
		while (getNs() < endNs);
	} else {
		// endless mode
		while (1);
	}

	return 0;
}
