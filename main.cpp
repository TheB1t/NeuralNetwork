#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include "NN.h"

double normalize(double min, double max, double value) {
	return (value - min) / (max - min);
}

double denormalize(double min, double max, double value) {
	return (value + min) * (max - min);
}

void setCursorPosition(int XPos, int YPos) {
    printf("\033[%d;%dH",YPos+1,XPos+1);
}

void getCursor(int* x, int* y) {
   printf("\033[6n");
   scanf("\033[%d;%dR", x, y);
}

int main() {
	srand(time(NULL));

	Network nn(2, { 8, 4, 8 }, 1);

	std::vector<std::vector<std::vector<double>>> dataSet {
		{ { 0, 0 }, { 0 } },
		{ { 1, 0 }, { 1 } },
		{ { 0, 1 }, { 1 } },
		{ { 1, 1 }, { 0 } }
	};

	double output[1] = { 0 };

	printf("\033[2J\033[1;1H");

#ifdef GNUPLOT_DATA
	FILE* graphFile = fopen("graphData.txt", "w");
#endif

	int i = 0;
	while (true) {
		setCursorPosition(0, 0);
		printf("Epoch %d\n", i++);
		
		double errSum = 0;
		for (int j = 0; j < dataSet.size(); j++) {
			nn.calc(dataSet[j][0].data(), output);
			nn.learn(dataSet[j][1].data());
			errSum += nn.getError();	
			printf("%-5.f XOR %-5.f = %-13.10f (%.5f) error %-6.10f\n", dataSet[j][0][0], dataSet[j][0][1], output[0], dataSet[j][1][0], nn.getError());
		}
		errSum /= dataSet.size();
		printf("Net Error %0.10f\n", errSum);

#ifdef GNUPLOT_DATA		
		fprintf(graphFile, "%d %f\n", i, errSum);
#endif

		if (errSum < 0.0000001) break;
	}

#ifdef GNUPLOT_DATA
	fclose(graphFile);
#endif

	for (int j = 0; j < dataSet.size(); j++) {
		nn.calc(dataSet[j][0].data(), output);
		printf("%-5.f XOR %-5.f = %-13.10f (%.5f) error %-6.10f\n", dataSet[j][0][0], dataSet[j][0][1], output[0], dataSet[j][1][0], nn.getError());
	}
};
