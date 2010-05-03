//===========================================================================
/*!
 *  \file simpleRNNet.cpp
 *
 *  \brief recurrent neural network example
 *  
 *
 *  \par Copyright (c) 1999-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#include <ReClaM/MSERNNet.h>
#include <ReClaM/Rprop.h>
#include <ReClaM/MeanSquaredError.h>
#include <ReClaM/createConnectionMatrix.h>
#include <Array/Array.h>
#include <Array/ArrayIo.h>


using namespace std;


// First we define a sub-class of the recurrent network
// to work with outputs in the range [-1:1], using the
// tanh activation function.
class MyRNNet : public MSERNNet
{
public:
	// Constructor for a network based on a given connection
	// matrix:
	MyRNNet( Array< int > con ) : MSERNNet( con ) { };

protected:
	// Define own activation function for the
	// hidden neurons, corresponding to the
	// prediction task (see below):
	double g( double a )
	{
		return tanh( a );
	}

	// A way to calculate the derivative of the
	// activation function, when given the result
	// of the activation:
	double dg( double ga )
	{
		return 1.0 - ga * ga;
	}
};


int main()
{
	// Just counter variables:
	unsigned i;

	// We will use 100 input patterns...
	unsigned numberOfDataPoints = 100;

	// ..., where 90 will be used for training
	// and the rest for initialization of the
	// network:
	unsigned warmUpLength = 10;

	// The number of training cycles:
	unsigned numberOfLearningCycles = 200;

	// Create connection matrix for two time steps and
	// 7 neurons:
	Array< int > con;

	// Creates a connection matrix for a recurrent network. 
	// We will use the network to predict the value for
	// simple time-series prediction.
	// Here we use a network with one input, 10 hidden,
	// and one output neuron (and one memory layer).
	createConnectionMatrixRNN(con, 1, 10, 1, 1);

	// Construct new network object on the base of the
	// connection matrix:
	MyRNNet net(con);

	//Construct Optimizer
	IRpropPlus optimizer;
	optimizer.init(net);

	// Initialize the weights of the neuron connections
	// by uniformally distributed random numbers between
	// "-0.1" and "0.1":
	net.initWeights(-0.1, 0.1);

	// Use the first ten input patterns for initializing
	// the internal state of the network:
	net.includeWarmUp(warmUpLength);

	// Create the data set:
	Array< double > input(numberOfDataPoints, 1),
	target(numberOfDataPoints, 1);

	// The task is to predict the next amplitude of a simple
	// oszillation:
	for (i = 0; i < input.dim(0); i++)
	{
		input(i, 0)  = sin(0.1 * i);
		target(i, 0) = sin(0.1 * (i + 1));
	}

	// Initialize the RProp optimization algorithm:
	optimizer.init(net);

	cout << "Train network:\n\n"
			 << "No. of cycles\tIteration Error:\n" 
			 << "--------------------------------" << endl;

	// Train the network by using the RProp algorithm,
	// monitor error each 10 training cycles:
	for (i = 1; i <= numberOfLearningCycles; i++)
	{
		optimizer.optimize(net, net, input, target);
		if (i % 10 == 0)
			cout << i << "\t\t" << net.error(net, input, target) << endl;
	}

	// Get the prediction results of the trained network:
	Array< double > output(target.dim(0), target.dim(1));

	net.model(input, output);

	cout << endl << "Evaluate trained network:\n\n"
			 << "x( t ):\t\tx( t + 1 ):\tprediction:\n" 
			 << "-------------------------------------------" << endl;

	// Output of each data input value, its value
	// in the next time step and the prediction of the
	// trained network:
	for (i = warmUpLength; i < target.dim(0); i++)
	{
		cout << input(i, 0) << "\t" << target(i, 0) << "\t"
		<< output(i, 0) << endl;
	}

	// lines below are for self-testing this example, please ignore
	if(net.error(net, input, target) < 0.00001) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

