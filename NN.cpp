#include "NN.h"

double activation(double x) {
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

double deactivation(double x) {
	return 1 - (x * x);
}

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

//------------------------------------------------------------------------------------------------------
//Layer class
Layer::Layer(int neuronCount): neuronCount(neuronCount) {
	this->neurons = new Neuron[neuronCount];
};

Layer::~Layer() {
/*
	for (int i = 0; i < this->size(); i++) {
		(*this)[i].linksNext.push_back(newLink);
		for (int j = 0; j < (*next)[j].linksPrev->size(); j++) {
			Link* newLink = new Link{ fRand(-0.5, 0.5), &(*this)[i], &(*next)[j] };
			
			(*next)[j].linksPrev.resize(0);
		};
	};
*/	
	delete this->neurons;
}

void Layer::bindNextLayer(Layer* next) {
	for (int i = 0; i < this->size(); i++) {
		for (int j = 0; j < next->size(); j++) {
			Link* newLink = new Link{ fRand(-0.5, 0.5), &(*this)[i], &(*next)[j] };
				
			(*this)[i].linksNext.push_back(newLink);
			(*next)[j].linksPrev.push_back(newLink);
		};
	};
};

int Layer::size() {
	return this->neuronCount;
};
		
Neuron& Layer::operator[](const int i) {
	return this->neurons[i];
};

const Neuron& Layer::operator[](const int i) const {
	return this->neurons[i];
};
				
void Layer::calc() {
	for (int i = 0; i < this->size(); i++) {
		double sum = 0;
		for (int j = 0; j < (*this)[i].linksPrev.size(); j++) {
			Link* current = (*this)[i].linksPrev[j];
			sum += current->from->output * current->weight;
		}

		(*this)[i].unactivatedOutput = sum;
		(*this)[i].output = activation(sum);
		//printf("Unactivated %.8f | Activated %.8f\n", sum, (*this)[i].output);
	}
			
	//printf("%s layer calculated\n", this->getType());	
};

void Layer::calcError() {
	for (int i = 0; i < this->size(); i++) {
		double sum = 0;
		for (int j = 0; j < (*this)[i].linksNext.size(); j++) {
			Link* current = (*this)[i].linksNext[j];
			sum += current->to->error * current->weight;
		}
		
		(*this)[i].error = sum * deactivation((*this)[i].unactivatedOutput);
	}					
}

void Layer::learn() {
	for (int i = 0; i < this->size(); i++) {
		for (int j = 0; j < (*this)[i].linksPrev.size(); j++) {
			Link* current = (*this)[i].linksPrev[j];
			double newWeight = current->weight + (LEARN_ALPHA * (*this)[i].error * current->from->output);
			if (newWeight < -1) newWeight = -1;
			if (newWeight > 1) newWeight = 1;
			//printf("Old weight %.8f | New weight %.8f\n", current->weight, newWeight);
			current->weight = newWeight;
		}
	}
}

//------------------------------------------------------------------------------------------------------
//InputLayer class
InputLayer::InputLayer(int neuronCount): Layer(neuronCount) {}

void InputLayer::setInputData(double* data) {
	for (int i = 0; i < this->size(); i++) {
		(*this)[i].output = data[i];
	}
};

void InputLayer::calc() {};
		
const char* InputLayer::getType() {
	return "Input";
};

//------------------------------------------------------------------------------------------------------
//MiddleLayer class
MiddleLayer::MiddleLayer(int neuronCount): Layer(neuronCount) {}
		
const char* MiddleLayer::getType() {
	return "Middle";
};

//------------------------------------------------------------------------------------------------------
//OutputLayer class
OutputLayer::OutputLayer(int neuronCount): Layer(neuronCount) {
	this->trainData = new double[neuronCount];
}

void OutputLayer::setTrainData(double* data) {
	memcpy(this->trainData, data, sizeof(double) * this->size());
}

void OutputLayer::getOutput(double* data) {
	for (int i = 0; i < this->size(); i++) {
		data[i] = (*this)[i].output;
	}
}

double OutputLayer::getMeanSquaredError() {
	return this->meanSquaredError;	
}

void OutputLayer::printOutput() {
	for (int i = 0; i < this->size(); i++) {
		printf("%d -> %.8f\n", i, (*this)[i].output);
	}	
}
		
void OutputLayer::calcError() {
	this->meanSquaredError = 0;
	for (int i = 0; i < this->size(); i++) {
		double tmp = this->trainData[i] - (*this)[i].output;
		(*this)[i].error = tmp * deactivation((*this)[i].output);
		this->meanSquaredError += tmp * tmp;
	}

	this->meanSquaredError *= 1 / this->size();
};
				
const char* OutputLayer::getType() {
	return "Output";
};

//------------------------------------------------------------------------------------------------------
//Network class
InputLayer* Network::getInputLayer() {
	return (InputLayer*)this->layers[0];
}

OutputLayer* Network::getOutputLayer() {
	return (OutputLayer*)this->layers[this->layers.size() - 1];
}

Network::Network(int INeurons, std::vector<int> MNeurons, int ONeurons) {
	this->layers.push_back(new InputLayer(INeurons));

	for (int i = 0; i < MNeurons.size(); i++) {
		this->layers.push_back(new MiddleLayer(MNeurons[i]));
	}

	this->layers.push_back(new OutputLayer(ONeurons));


	for (int i = 0; i < this->layers.size() - 1; i++) {
		this->layers[i]->bindNextLayer(this->layers[i + 1]);
	}
}

void Network::calc(double* data, double* output) {
	this->getInputLayer()->setInputData(data);
	
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers[i]->calc();
	}

	this->getOutputLayer()->getOutput(output);
}

void Network::learn(double* data) {
	this->getOutputLayer()->setTrainData(data);
		
	for (int i = this->layers.size() - 1; i >= 0; i--) {
		this->layers[i]->calcError();
	}
			
	for (int i = 1; i < this->layers.size(); i++) {
		this->layers[i]->learn();
	}
}

double Network::getError() {
	return this->getOutputLayer()->getMeanSquaredError();
}
