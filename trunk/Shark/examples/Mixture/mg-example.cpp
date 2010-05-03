//===========================================================================
/*!
 *  \file mg-example.cpp
 *
 *
 *  \par Copyright (c) 1999-2003:
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
//#include "ReClaM/BFGS.h"
//#include "ReClaM/Rprop.h"
// #include "Mixture/RBFN.h"
//#include "ReClaM/SquaredError.h"
//#include "ReClaM/ErrorMeasures.h"
#include <Rng/RNG.h>
#include <cstdio>

using namespace std;

//===========================================================================

int main(int argc, char **argv)
{
	const unsigned EmbedDim = 5;
	const unsigned TimeLag  = 1;
	const unsigned Horizon  = 1;
	const unsigned NumSkip  = 500;
	const unsigned NumTrain = 1000;
	const unsigned NumTest  = 1000;
	//const unsigned NumIter  = 1000;

	unsigned i, j;
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
		ArrayReference< double > out(outTrain[ 0 ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}
	for (i = 0; i < NumTrain; ++i) {
		ArrayReference< double > in(inTrain [ i ]);
		ArrayReference< double > out(outTrain[ i ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}
	for (i = 0; i < NumTest; ++i) {
		ArrayReference< double > in(inTest [ i ]);
		ArrayReference< double > out(outTest[ i ]);
		iosamples(static_cast< Array< double >& >(in),
				  static_cast< Array< double >& >(out));
	}

	char s[ 100 ];
	for (j = 0; j < inTrain.dim(1); ++j) {
		if (j)
			cout << ',';
		sprintf(s, "MG[%d]", inTrain.dim(1) - j - 1);
		cout << s;
	}
	for (j = 0; j < outTrain.dim(1); ++j) {
		sprintf(s, ",MG[%d]", j + 1);
		cout << s;
	}
	cout << endl;
	for (i = 0; i < inTrain.dim(0); ++i) {
		for (j = 0; j < inTrain.dim(1); ++j) {
			if (j)
				cout << ',';
			cout << inTrain(i, j);
		}
		for (j = 0; j < outTrain.dim(1); ++j)
			cout << ',' << outTrain(i, j);
		cout << endl;
	}
	return 0;
}

//===========================================================================

