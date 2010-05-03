/*!
*  \file MSERNNet.cpp
*
*  \brief Mean Squared Error Recurrent Neural Network
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


#include <SharkDefs.h>
#include <ReClaM/MSERNNet.h>
#include <sstream>


using namespace std;


MSERNNet::MSERNNet()
{
	init0();
}

MSERNNet::MSERNNet(Array<int> con)
{
	init0();
	setStructure(con);
}

MSERNNet::MSERNNet(Array<int> con, Array<double> wei)
{
	init0();
	setStructure(con, wei);
}

MSERNNet::MSERNNet(std::string filename)
{
	init0();
	setStructure(filename);
}


void MSERNNet::includeWarmUp(unsigned WUP)
{
	warmUpLength = WUP;
}

void MSERNNet::model(const Array<double> &input)
{
	if (input.ndim() != 2)
	{
		throw SHARKEXCEPTION("STOP: MSERNNet::model requires a 2-dim input array");
	}
	if (warmUpLength > 0) Y = 0.;
	processTimeSeries(input);
}

void MSERNNet::model(const Array<double>& input, Array<double>& output)
{
	if (output.ndim() != 2)
	{
		throw SHARKEXCEPTION("STOP: MSERNNet::model requires a 2-dim output array");
	}
	unsigned t, i, T = output.dim(0), M = output.dim(1);
	model(input);
	for (t = 0;t < T;t++)
		for (i = 0;i < M;i++)
			output(t, i) = Y(numberOfNeurons - 1 - i, T - 1 - t);
}

double MSERNNet::error(Model& model, const Array<double> &input, const Array<double> &target)
{
	if (target.ndim() != 2)
	{
		throw SHARKEXCEPTION("STOP: MSERNNet::error requires a 2-dim target array");
	}

	unsigned i, t, T, M;
	T = target.dim(0);
	M = target.dim(1);

	if (T <= warmUpLength) throw SHARKEXCEPTION("[MSERNNet::error] number of patterns not larger than warmup length");
	this->model(input);

	err = 0;
	double totalError = 0;
	double z;
	for (t = 0;t < warmUpLength;t++)
	{
		for (i = 0;i < M;i++)
		{
			err(numberOfNeurons - 1 - i, T - 1 - t) = 0.0;
		}
	}
	for (t = warmUpLength;t < T;t++)
	{
		for (i = 0;i < M;i++)
		{
			z = err(numberOfNeurons - 1 - i, T - 1 - t) = Y(numberOfNeurons - 1 - i, T - 1 - t) - target(t, i);
			totalError += z * z;
		}
	}
	totalError /= (T - warmUpLength) * M;
	return totalError;
}

double MSERNNet::errorPercentage(const Array<double> &input, const Array<double> &target)
{
	unsigned t;
	double outmax = -MAXDOUBLE;
	double outmin = MAXDOUBLE;

	if (target.ndim() != 2)
	{
		throw SHARKEXCEPTION("STOP: MSERNNet::error requires a 2-dim target array");
	}

	unsigned T, M;
	T = target.dim(0);
	M = target.dim(1);

	if (T <= warmUpLength)
	{
		throw SHARKEXCEPTION("number of patterns not larger than warmup length");
	}

	this->model(input);

	err      = 0;
	double totalError = 0;
	double z;
	for (t = 0; t < warmUpLength; t++)
		for (unsigned i = 0; i < M; i++)
		{
			err(numberOfNeurons - 1 - i, T - 1 - t) = 0.0;
		}
	for (t = warmUpLength; t < T; t++)
		for (unsigned i = 0; i < M; i++)
		{
			if (target(t, i) > outmax) outmax = target(t, i);
			if (target(t, i) < outmin) outmin = target(t, i);
			z = err(numberOfNeurons - 1 - i, T - 1 - t) = Y(numberOfNeurons - 1 - i, T - 1 - t) - target(t, i);
			totalError += z * z;
		}
	totalError /= (T - warmUpLength) * M;
	totalError *= 100 / ((outmax - outmin) * (outmax - outmin));
	return totalError;
}

double MSERNNet::meanSquaredError(const Array<double> &input,
								  const Array<double> &target)
{
	return error(*this, input, target);
}

double MSERNNet::errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative)
{
	double e;
	e = error(model, input, target);
	calcGradBPTT();

	writeGradient(derivative);
	derivative /= (double)(target.dim(0) - warmUpLength) * target.dim(1);
	return e;
}

Array<int> & MSERNNet::getConnections()
{
	return connectionMatrix.A;
}

Array<double> & MSERNNet::getWeights()
{
	return weightMatrix.A;
}

void MSERNNet::initWeights(long seed, double min, double max)
{
	Rng::seed(seed);
	initWeights(min, max);
}

void MSERNNet::initWeights(double low, double up)
{
	unsigned i, j, t;
	for (t = 0;t < delay;t++) for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < numberOfNeurons;j++)
			{
				if (connectionMatrix(t, i, j)) weightMatrix(t, i, j) = Rng::uni(low, up);
				else weightMatrix(t, i, j) = 0;
			}
	writeParameters();
}

void MSERNNet::setStructure(Array<int> &mat)
{
	unsigned i, j, t;

	delay = mat.dim(0);
	numberOfNeurons = mat.dim(1);
	weightMatrix.resize(delay, numberOfNeurons, numberOfNeurons, false);
	dEw.resize(delay, numberOfNeurons, numberOfNeurons, false);
	stimulus.resize(numberOfNeurons, false);
	connectionMatrix.resize(delay, numberOfNeurons, numberOfNeurons, false);
	delMask.resize(delay, false);

	Y.resize(numberOfNeurons, delay, false);
	Y = 0;
	delta.resize(numberOfNeurons, delay, false);
	delta = 0;

	connectionMatrix.A = mat;
	numberOfParameters = 0;
	for (t = 0;t < delay;t++)
	{
		delMask(t) = 0;
		for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < numberOfNeurons;j++) if (connectionMatrix(t, i, j))
				{
					numberOfParameters++;
					delMask(t) = 1;
				}
	}
	parameter.resize(numberOfParameters, false);
}

void MSERNNet::setStructure(Array<int> &con, Array<double> &wei)
{
	setStructure(con);
	setStructure(wei);
}

void MSERNNet::setStructure(Array<double> &wei)
{
	weightMatrix.A = wei;
	writeParameters();
}

void MSERNNet::setStructure(std::string filename)
{
	std::ifstream is;
	is.open(filename.c_str());
	unsigned nDelay;
	is >> nDelay;

	std::vector<int> l;
	const unsigned Len = 256;
	char buffer[Len];
	unsigned n = 0;
	is >> std::setw(Len) >> buffer;

	while (is)
	{
		n++;
		l.push_back(atoi(buffer));
		if (is.peek() == '\n')
		{
			break;
		}

		is >> std::setw(Len) >> buffer;
	}

	if (n == 0)
	{
		throw SHARKEXCEPTION("[MSERNNet::setStructure] no matrices given in net file");
	}

	Array<int> con(nDelay, n, n);
	Array<double> wei(nDelay, n, n);
	unsigned i, j, k;

	for (i = 0; i < con.dim(0); i++)
		for (j = 0; j < con.dim(1); j++)
			if ((i == 0) && (j == 0))
			{
				for (k = 0; k < n; k++) con(i, j, k) = l[k];
			}
			else
				for (k = 0; k < con.dim(2); k++)
				{
					is >> con(i, j, k);
					if (!is)
					{
						throw SHARKEXCEPTION("[MSERNNet::setStructure] not enough entries to build connection matrix");
					}
				}

	for (i = 0; i < wei.dim(0); i++)
		for (j = 0; j < wei.dim(1); j++)
			for (k = 0; k < wei.dim(2); k++)
			{
				is >> wei(i, j, k);
				if (!is)
				{
					throw SHARKEXCEPTION("[MSERNNet::setStructure] not enough entries to build weight matrix");
				}
			}

	setStructure(con, wei);
}

void MSERNNet::write(std::string filename)
{
	unsigned i, j, k;
	FILE *fp = fopen(filename.c_str(), "wt");

	Array<int> con = getConnections();
	unsigned nDelay = con.dim(0);
	fprintf(fp, "%u\n", nDelay);

	for (k = 0; k < nDelay; k++)
	{
		for (i = 0; i < con.dim(1); i++)
		{
			for (j = 0; j < con.dim(2); j++)
			{
				fprintf(fp, "%1d", con(k, i, j));
				if (j < con.dim(2) - 1) fprintf(fp, " ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}

	Array<double> wei = getWeights();
	for (k = 0; k < nDelay; k++)
	{
		for (i = 0; i < wei.dim(1); i++)
		{
			for (j = 0; j < wei.dim(2); j++)
			{
				fprintf(fp, "%10.8e", wei(k, i, j));
				if (j < wei.dim(2) - 1) fprintf(fp, " ");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void MSERNNet::setHistory(const Array<double> &Ystate)
{
	if (Ystate.dim(0) != numberOfNeurons || Ystate.dim(1) != delay)
	{
		throw SHARKEXCEPTION("STOP: MSERNNet::setHistory: wrong dimensions");
	}
	unsigned i, j;
	for (i = 0;i < numberOfNeurons;i++)
		for (j = 0;j < delay;j++)
			Y(i, j) = Ystate(i, j);
}

Array<double> MSERNNet::getHistory()
{
	unsigned i, j;
	Array<double> Ystate(numberOfNeurons, delay);
	for (i = 0;i < numberOfNeurons;i++)
		for (j = 0;j < delay;j++)
			Ystate(i, j) = Y(i, j);
	return Ystate;
}


//###########################################################################
//
// Protected Methods:
//
//###########################################################################


// Activation function of all neurons.
double MSERNNet::g(double a)
{
	return 1 / (1 + exp(-a));
}


// Computes the derivative of g(a) for all neurons.
double MSERNNet::dg(double ga)
{
	return ga*(1 - ga);
}


// Initializes some internal variables.
void MSERNNet::init0()
{
	time = numberOfNeurons = delay = episode = numberOfParameters = 0;
	warmUpLength = 0;
	Y.resize(0, 0, false);
	err.resize(0, 0, false);
	delta.resize(0, 0, false);
}


// Writes the values of the weights stored in the weight matrix to
// the parameter vector of the model.
void MSERNNet::writeParameters()
{
	unsigned i, j, t, n = 0;
	for (t = 0;t < delay;t++) for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < numberOfNeurons;j++)
				if (connectionMatrix(t, i, j))	parameter(n++) = weightMatrix(t, i, j);
}


// Reads the values of the parameter vector parameter of Model
// and stores these values in the weight matrix.
void MSERNNet::readParameters()
{
	unsigned i, j, t, n = 0;
	for (t = 0;t < delay;t++) for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < numberOfNeurons;j++)
							if (connectionMatrix(t, i, j)) weightMatrix(t, i, j) = parameter(n++); else weightMatrix(t, i, j) = 0;
}


// Writes the values of the gradient of the error with respect to the
// different weights from the variable dEw to the parameter derivative
void MSERNNet::writeGradient(Array<double>& derivative)
{
	unsigned i, j, t, n = 0;
	derivative.resize(numberOfParameters, false);
	for (t = 0;t < delay;t++) for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < numberOfNeurons;j++)
				if (connectionMatrix(t, i, j)) derivative(n++) = dEw(t, i, j);
}


// Performs some initializations that are necessary to process the
// time series.
void MSERNNet::prepareTime(unsigned t)
{
	unsigned i, j;
	ArrayTable<double> Ystate, Dstate;
	Ystate.resize(numberOfNeurons, delay, false);
	Dstate.resize(numberOfNeurons, delay, false);
	for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < delay;j++)
		{
			Ystate(i, j) = Y(i, j);
			Dstate(i, j) = delta(i, j);
		}
	if (t != episode)
	{
		episode = t;
		Y.resize(numberOfNeurons, episode + delay, false);
		Y = 0;
		delta.resize(numberOfNeurons, episode + delay, false);
		delta = 0;
		err.resize(numberOfNeurons, episode + delay, false);
		err = 0;
	}
	for (i = 0;i < numberOfNeurons;i++) for (j = 0;j < delay;j++)
		{
			Y(i, t + j) = Ystate(i, j);
			delta(i, t + j) = Dstate(i, j);
		}
	time = t;
}


// Processes a whole time series. After processing the output can be
// found in the variable Y.
void MSERNNet::processTimeSeries(const Array<double>& input)
{
	unsigned t, i;
	if (input.dim(1) >= numberOfNeurons)
	{
		throw SHARKEXCEPTION("[MSERNNet::processTimeSeries] structure has not enough neurons to handle input - use setStructure(...)");
	}

	readParameters();
	prepareTime(input.dim(0));
	stimulus = 0;
	for (t = 0;t < input.dim(0);t++)
	{
		for (i = 0;i < input.dim(1);i++) stimulus(i) = input(t, i);
		processTimeStep();
	}
}


// Processes one input pattern.
void MSERNNet::processTimeStep()
{
	unsigned i, j, t;
	double z;
	if (!time)
	{
		throw SHARKEXCEPTION("[MSERNNet::process] no time prepared!");
	}
	time--;
	for (i = 0;i < numberOfNeurons;i++)
	{
		z = stimulus(i);
		z += weightMatrix(0, i, i);
		if (delMask(0))
			for (j = 0;j < i;j++)
				z += weightMatrix(0, i, j) * Y(j, time);
		for (t = 1;t < delay;t++)
			if (delMask(t))
				for (j = 0;j < numberOfNeurons;j++)
					z += weightMatrix(t, i, j) * Y(j, time + t);
		Y(i, time) = g(z);
	}
}


// Performs backpropagation through time to calculate the derivative
// of the error with respect to the weights. The results are stored to
// dEw.
void MSERNNet::calcGradBPTT()
{
	int i, j, t, s, n;
	double z;

	for (t = 0;t < (int)episode;t++) for (i = numberOfNeurons - 1;i >= 0;i--)
		{
			z = err(i, t);
			if (delMask(0)) for (j = numberOfNeurons - 1;j > i;j--)
					z += delta(j, t) * weightMatrix(0, j, i);
			for (s = 1;s < (int)delay && s <= t;s++) if (delMask(s)) for (j = numberOfNeurons - 1;j >= 0;j--)
						z += delta(j, t - s) * weightMatrix(s, j, i);
			delta(i, t) = dg(Y(i, t)) * z;
		}

	for (t = 0;t < (int)delay;t++) if (delMask(t)) for (i = 0;i < (int)numberOfNeurons;i++)
			{
				if (t > 0) n = numberOfNeurons; else n = i;
				for (j = 0;j < n;j++) if (connectionMatrix(t, i, j))
					{
						dEw(t, i, j) = 0;
						for (s = 0;s + t < (int)episode;s++) dEw(t, i, j) += delta(i, s) * Y(j, t + s);
						dEw(t, i, j) *= 2.0;
					}
			}

	for (i = 0;i < (int)numberOfNeurons;i++) if (connectionMatrix(0, i, i))
		{
			dEw(0, i, i) = 0;
			for (s = 0;s < (int)episode;s++) dEw(0, i, i) += delta(i, s);
			dEw(0, i, i) *= 2.0;
		}
}

//===========================================================================
/*!
 *  \brief Creates a connection matrix for a recurrent network.
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void MSERNNet::setStructure(unsigned in, unsigned hidden, unsigned out,
							   unsigned memory, bool layered,
							   bool recurrentInputs, bool bias, bool elman, bool previousInputs)
{
	unsigned N;  // total number of neurons:
	unsigned k, i, j;
	Array<int> con;

	//
	// Calculate total number of neurons from the
	// number of neurons per layer:
	//
	N = in + hidden + out;
	con.resize(1 + memory, N, N);
	con[0] = 0;

	if (!layered)
	{
		for (i = in; i < N; i++)
		{
			for (j = 0; j < i; j++)
				con(0, i, j) = 1;
			if (bias) con(0, i, i) = 1;
		}
	}
	else
	{
		for (i = in; i < N - out; i++)
		{
			for (j = 0; j < in; j++)
				con(0, i, j) = 1;
			if (bias) con(0, i, i) = 1;
		}
		if (elman)
			for (i = N - out; i < N; i++)
			{
				for (j = 0; j < N - out; j++)
					con(0, i, j) = 1;
				if (bias) con(0, i, i) = 1;
			}
		else
			for (i = N - out; i < N; i++)
			{
				for (j = 0; j < in; j++)
					con(0, i, j) = 1;
				if (bias) con(0, i, i) = 1;
			}
	}
	if (recurrentInputs)
		for (i = 1; i <= memory; i++) con[i] = 1;
	else
	{
		for (i = 1; i <= memory; i++)
		{
			con[i] = 1;
			for (j = 0; j < in; j++)
			{
				(con[i])[j] = 0;
				if (!previousInputs) for (k = in; k < N; k++) con(i, k, j) = 0;
			}
		}
	}
	setStructure(con);
}
