#pragma once
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <sstream>
#include <fstream>

#ifndef LEARN_ALPHA
	#define LEARN_ALPHA 0.0025
#endif

struct Link {
	double	weight			= 0;
	struct	Neuron* from	= 0;
	struct	Neuron* to		= 0;
};

struct Neuron {
	std::vector<Link*>	linksPrev;
	std::vector<Link*>	linksNext;
	double				unactivatedOutput	= 0;
	double				output				= 0;
	double				error				= 0;
};

class Layer {
	protected:
		int		neuronCount	= 0;
		Neuron*	neurons		= 0;
		
		Layer(int neuronCount);
		~Layer();
		
	public:
		void bindNextLayer(Layer* next);
		int size();
		
		Neuron& operator[](const int i);
		const Neuron& operator[](const int i) const;
		
		virtual void calc();
		virtual void calcError();
		virtual void learn();
		virtual const char* getType() = 0;
};

class InputLayer: public Layer {
	public:
		InputLayer(int neuronCount);

		void setInputData(double* data);

		virtual void calc();
		virtual const char* getType();
};

class MiddleLayer: public Layer {
	public:
		MiddleLayer(int neuronCount);
		
		virtual const char* getType();
};

class OutputLayer: public Layer {
	private:
		double* trainData;
		double meanSquaredError = 0;
		
	public:
		OutputLayer(int neuronCount);

		void setTrainData(double* data);
		void getOutput(double* data);
		double getMeanSquaredError();
		void printOutput();
		
		virtual void calcError();
		virtual const char* getType();
};

class Network {
	private:
		std::vector<Layer*> layers;

		InputLayer* getInputLayer();
		OutputLayer* getOutputLayer();
		
	public:
		Network(int INeurons, std::vector<int> MNeurons, int ONeurons);

		void calc(double* data, double* output);
		void learn(double* data);
		double getError();
};
