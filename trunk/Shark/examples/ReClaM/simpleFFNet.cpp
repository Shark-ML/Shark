//===========================================================================
/*!
 *  \file simpleFFNet.cpp
 *
 *  \brief feed forward neural network example
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
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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
#include <Array/Array.h>
#include <ReClaM/LinOutFFNet.h>
#include <ReClaM/CG.h>
#include <ReClaM/Rprop.h>
#include <ReClaM/MeanSquaredError.h>
#include <ReClaM/createConnectionMatrix.h>
#include <Rng/GlobalRng.h>
#include <iostream>


using namespace std;


int main()
{
	// just counter variables:
	unsigned i;
	
	unsigned numberOfHiddenNeurons = 25;
	unsigned numberOfLearningCycles = 500;
	
	// Create a connection matrix with 2 inputs, 1 output
	// and a single, fully connected hidden layer with
	// numberOfHiddenNeurons neurons:
	Array<int> con;
	createConnectionMatrix(con, 2, numberOfHiddenNeurons, 1);
	
	// Construct two new network objects on the base of the
	// connection matrix. The first one will be used for
	// training, the second one will store the network
	// with the smallest validation error:
	LinOutFFNet net(2, 1, con), netMin(2, 1, con);
	// construct Error
	MeanSquaredError error;
	// construct Optimizer
	CG optimizer;
	// IRpropPlus optimizer;		// (better) alternative
	optimizer.init(net);


	// Initialize the weights of the neuron connections
	// by uniformally distributed random numbers between
	// "-0.1" and "0.1":
	net.initWeights(-0.1, 0.1);
	
	// The untrained network is the best one until now.
	netMin = net;

	// Definition of training and validation data
	unsigned int examples = 100;
	double stddev = 1.0;
	Array< double > inputTrain(examples, 2u), targetTrain(examples, 1u);
	Array< double > inputValidate(examples, 2u), targetValidate(examples, 1u);

	// Construct data:
	for (i = 0; i < examples; i++) 
	{
		inputTrain(i, 0) = Rng::uni(-5.0, 5.0);
		inputTrain(i, 1) = Rng::uni(-5.0, 5.0);

		inputValidate(i, 0) = Rng::uni(-5.0, 5.0);
		inputValidate(i, 1) = Rng::uni(-5.0, 5.0);

		targetTrain(i, 0) = inputTrain(i, 0)*inputTrain(i, 0) + inputTrain(i, 1)*inputTrain(i, 1) + stddev * Rng::gauss();

		targetValidate(i, 0) = inputValidate(i, 0)*inputValidate(i, 0) + inputValidate(i, 1)*inputValidate(i, 1) + stddev * Rng::gauss();
	}	

	// Current error of the training set:
	double t;

	// Current error of the validation set:
	double v;

	// The smallest error for the validation set:
	double vMin = error.error(net, inputValidate, targetValidate);
	
	// The training epoch, where the network with the
	// smallest error percentage for the validation set
	// occured:
	unsigned epochMin = 0;


	cout << "Train network:\n" << endl
			 << "epoch:\ttraining error:\tvalidation error:" << endl
			 << "-----------------------------------------" << endl;
	
	// Training of the network:
	for (unsigned epoch = 1; epoch <= numberOfLearningCycles; epoch++)
		{
			
			// Train the net with conjugate gradients ...
			optimizer.optimize(net, error , inputTrain, targetTrain);
			
			//  ... and calculate the (monitoring) errors:
			t = error.error(net, inputTrain, targetTrain);
			v = error.error(net, inputValidate, targetValidate);
			
			// Monitor the results:
			std::cout << epoch << "\t  " << t << "\t\t" << v << endl;
			
			// Memorize the network with smallest validation error:
			if (v < vMin)
				{
					vMin = v;
					netMin = net;
					epochMin = epoch;
				}
		}
	
	//  Output of the performance values of the best network:
	t = error.error(netMin, inputTrain, targetTrain);
	v = error.error(netMin, inputValidate, targetValidate);
	
	cout << "\n\n\nError of network with best validation error:\n\n"
			 << "epoch:\ttraining error:\tvalidation error:\n"
			 << "-----------------------------------------\n" 
			 << epochMin << "\t" << t << "\t" << v << endl;
	
	//  Output of the structure of the best network:
// 	cout << "\n\n\nStructure of this network\n(no. of input neurons, "
// 			 << "no. of output neurons, connection matrix, weight matrix):\n"
// 			 << endl;
// 	cout.precision(6);
// 	cout.setf(ios::fixed | ios::showpos);
// 	cout << netMin << endl;
	
	// Show network behaviour for one input pattern:
	cout << "Processing a single input pattern:\n" << endl;
	Array< double > in(2), out(1);
	in(0) = 0.3;
	in(1) = -0.1;
	netMin.model(in, out);
	cout << "Input:\t( " << in(0) << ", " << in(1) << " ) " << endl;
	cout << "Output:\t" << out(0) << endl;	

	// lines below are for self-testing this example, please ignore
	if(error.error(netMin, inputTrain, targetTrain) < 0.5) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
