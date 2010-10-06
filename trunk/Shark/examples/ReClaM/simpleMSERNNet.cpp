//===========================================================================
/*!
 *  \file simpleMSERNNet.cpp
 *
 *  \brief example for a recurrent neural network, that is combined with the mean squared error measure
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

#include<ReClaM/MSERNNet.h>
#include<ReClaM/Rprop.h>
#include<ReClaM/createConnectionMatrix.h>
#include<Array/ArrayIo.h>

using namespace std;

//===========================================================================
// define the regression model

class myNet: public MSERNNet, public RpropPlus
{
public:
	myNet(): MSERNNet()
	{
		Array <int> con;//
		createConnectionMatrixRNN(con,1,8,1,2);
		setStructure(con);
		initWeights(-0.5, 0.5);
		init(*this);
	};

//   Array<double> &Dedw(){ return dedw; }
};


//===========================================================================
// learn the Lorentz time series

int main(int argc, char *argv[])
{

	//===========================================================================
	// some parameters

	unsigned iterations = 1000;
	const char* datafile = "timeseries";
	if(argc >1) datafile = argv[1];
	
	unsigned episode = 1000;
	int forecast = 5;
	bool checkGradient = false;     //if this is true, the gradient calculation is checked

	unsigned long t = 0, i;
	ifstream infile;
	ofstream outfile;

	//load the data
	Array<double> data;
	infile.open(datafile);
	readArray(data, infile);
	infile.close();
	data /= 100.; data += .5; //scale between [0,1] because all neurons of a RNN are sigmoid!

	//define the training and evaluation arrays
	unsigned dims[2] = {episode, 1};
	ArrayReference<double> trainIn(dims, &data.elem(0)            , 2, episode);
	ArrayReference<double> trainTarget(dims, &data.elem(0)   + forecast, 2, episode);
	ArrayReference<double> evalIn(dims, &data.elem(1000)         , 2, episode);
	ArrayReference<double> evalTarget(dims, &data.elem(1000) + forecast, 2, episode);
	Array<double> trainOut(episode, 1), evalOut(episode, 1);

	//output the evaluation input/target arrays -- to compare them to evalOut
	outfile.open("input");
	writeArray(evalIn, outfile);
	outfile.close();
	outfile.open("target");
	writeArray(evalTarget, outfile);
	outfile.close();

	//create the network
	myNet net;
	Array<double> exact, estimated;
	double z;

	while (t++ < iterations)
	{
		cout << t << "\t";

		//WARNING!!: for a unique error, warmupLength must be >0
		// -- only then, the neuron states are reset to zero
		// in the model routine
		net.includeWarmUp(100);

		//check gradient on the training data
		if (checkGradient)
		{
			net.errorDerivative(net, trainIn, trainTarget, exact);
			cout << "derror - gradient:" << exact;
			net.ErrorFunction::errorDerivative(net, trainIn, trainTarget, estimated);
			cout << "deltagrad - gradient:" << estimated();
			for (z = 0, i = 0;i < exact.nelem();i++)
			{
				exact(i) = exact(i) - estimated(i); z += exact(i) * exact(i);
			}
			cout << "difference:" << exact;
			cout << "square-sum of difference: " << z << "\n";
		}

		//learn on the training data
		net.optimize(net, net, trainIn, trainTarget);

		//evaluate on the evaluation data (compare files `output' `input' `target')
		cout << net.error(net, evalIn, evalTarget) << endl;
		if (!(t % 100))
		{
			net.model(evalIn, evalOut);
			outfile.open("output");
			writeArray(evalOut, outfile);
			outfile.close();
		}
	}

	// lines below are for self-testing this example, please ignore
	if (net.error(net, evalIn, evalTarget) < 0.00135) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
