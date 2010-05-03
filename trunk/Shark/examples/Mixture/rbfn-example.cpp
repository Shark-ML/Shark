//===========================================================================
/*!
 *  \file rbfn-example.cpp
 *
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      Mixture
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Mixture. This library is free software;
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



#include <Array/ArrayIo.h>
#include <TimeSeries/DiscreteMackeyGlass.h>
#include <TimeSeries/IOSamples.h>
#include <ReClaM/Model.h>
#include <ReClaM/MeanSquaredError.h>
#include <ReClaM/Rprop.h>
// #include <ReClaM/BFGS.h>
#include <Mixture/RBFN.h>
#include <Rng/RNG.h>
#include <iomanip>



class RBFN_Model : public RBFN, virtual public Model
{
public:
	RBFN_Model(unsigned numInput, unsigned numOutput, unsigned numHidden)
			: RBFN(numInput, numOutput, numHidden)
	{
		parameter.resize(b.nelem(), false);
		getParams(parameter);
	}

	void initialize(const Array< double >& in, const Array< double >& out)
	{
		RBFN::initialize(in, out);
		getParams(parameter);
	}

	void model(const Array<double>& in, Array<double>& out)
	{
		setParams(parameter);
		recall(in, out);
	}

	void modelDerivative(const Array<double>& in, Array<double>& derivative)
	{
		setParams(parameter);
		gradientOut(in, derivative);
	}

	void modelDerivative(const Array<double>& in, Array<double>& out, Array<double>& derivative)
	{
		setParams(parameter);
		recall(in, out);
		gradientOut(in, derivative);
	}

	void setParameter(unsigned int index, double value)
	{
		Model::setParameter(index, value);
		setParams(parameter);
	}
};


int main(int argc, char **argv)
{
	const unsigned EmbedDim = 5;
	const unsigned TimeLag  = 1;
	const unsigned Horizon  = 1;
	const unsigned NumSkip  = 500;
	const unsigned NumTrain = 1000;
	const unsigned NumTest  = 1000;
	const unsigned NumIter  = 20;

	unsigned i, t;
	Array< double > inTrain(NumTrain, EmbedDim);
	Array< double > outTrain(NumTrain, 1);
	Array< double > inTest(NumTest,  EmbedDim);
	Array< double > outTest(NumTest,  1);

	DiscreteMackeyGlass mg;
	IOSamples< double > iosamples(mg, EmbedDim, TimeLag, 1, Horizon);

	//
	// create samples for training and test
	//
	for (i = 0; i < NumSkip; ++i) {
		ArrayReference< double > in(inTrain [ 0 ]);
		ArrayReference< double > out(outTrain [ 0 ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}
	for (i = 0; i < NumTrain; ++i) {
		ArrayReference< double > in(inTrain [ i ]);
		ArrayReference< double > out(outTrain [ i ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}
	for (i = 0; i < NumTest; ++i) {
		ArrayReference< double > in(inTest [ i ]);
		ArrayReference< double > out(outTest [ i ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}

	RNG::globalRng.seed(1);

	RBFN_Model rbfn1(EmbedDim, 1, 10);
	RBFN_Model rbfn2(EmbedDim, 1, 10);
	MeanSquaredError mse;
	IRpropPlus rprop1;
	IRpropMinus rprop2;

	rbfn1.initialize(inTrain, outTrain);
	rbfn2.initialize(inTrain, outTrain);
	rprop1.initUserDefined(rbfn1, argc > 1 ? atof(argv[ 1 ]) : 0.0125);
	rprop2.initUserDefined(rbfn1, argc > 1 ? atof(argv[ 1 ]) : 0.0125);

	for (t = 0; t < NumIter; ++t) {
		rprop1.optimize(rbfn1, mse, inTrain, outTrain);
		std::cout << t << '\t'
		<< mse.error(rbfn1, inTrain, outTrain) << '\t'
		<< mse.error(rbfn1, inTest, outTest) << '\t';
		rprop2.optimize(rbfn2, mse, inTrain, outTrain);
		std::cout << mse.error(rbfn2, inTrain, outTrain) << '\t'
		<< mse.error(rbfn2, inTest, outTest) << std::endl;
	}
	// lines below are for self-testing this example, please ignore
       	if( mse.error(rbfn1, inTrain, outTrain)<0.0594735) exit(EXIT_SUCCESS);
       	else exit(EXIT_FAILURE);
}

