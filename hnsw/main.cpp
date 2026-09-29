#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void sift_test1B(char*, char*, int, int, int, int, float, int);
int main(int argc, char **argv) {
	if (argc != 9) {
		fprintf(stderr, "Usage: %s data queries n dim T qn delta_or_k efsearch; "
		        "T uses the paper's finest quantization level in [4,7]\n", argv[0]);
		return 1;
	}
	char  data_set[200];
	int vecsize_;
    int vecdim_;
    int level_;	
	float delta_;
	int qsize_;
	int efsearch_;
	
	//strncpy(data_set, argv[1], sizeof(data_set));
	vecsize_ = atoi(argv[3]);
	vecdim_ = atoi(argv[4]);
	const int paper_T = atoi(argv[5]);
	if (paper_T < 4 || paper_T > 7) {
		fprintf(stderr, "Invalid HVS T: require paper-level T in [4,7]\n");
		return 1;
	}
	level_ = paper_T - 3;
	qsize_ = atoi(argv[6]);
	delta_ = atof(argv[7]);
	efsearch_ = atoi(argv[8]);
	
    sift_test1B(argv[1], argv[2], vecsize_, vecdim_, level_, qsize_, delta_, efsearch_);

    return 0;
};
