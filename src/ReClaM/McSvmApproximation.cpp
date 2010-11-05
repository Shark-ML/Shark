//===========================================================================
/*!
 *  \file McSvmApproximation.cpp
 *
 *  \author  B. Weghenkel
 *  \date    2010
 *
 *  \brief Approximation of Multi-class Support Vector Machines (SVMs)
 *
 *  \par Copyright (c) 2006:
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


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include <stdlib.h>
#include <math.h>

#include <SharkDefs.h>
#include <Rng/Uniform.h>
#include <Rng/DiscreteUniform.h>
#include <Rng/GlobalRng.h>
#include <Array/Array2D.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <ReClaM/Rprop.h>
#include <ReClaM/McSvmApproximation.h>
#include <ReClaM/KernelKMeans.h>


using namespace std;


McSvmApproximation::McSvmApproximation(MultiClassSVM *pSVM, bool verbose)
{
	this->verbose = verbose;

	unsigned int c, t;

	mpSVM             = pSVM;  // store pointer to original SVM
	mpApproximatedSVM = NULL;

	// get parameters of original SVM
	mpKernel             = mpSVM->getKernel();
	mGamma               = mpKernel->getParameter(0);
	mDimension           = mpSVM->getDimension();
	mNoClasses			 = mpSVM->getClasses();
	mNoExamplesOfOrigSVM = mpSVM->getExamples();   // include those examples with \alpha_i = 0

	labels = new double[mNoExamplesOfOrigSVM];

	// determine number of SVs of original SVM
	mNoUniqueSVs = 0;
	mNoNonUniqueSVs = 0;
	mpNoSVsOrigSVM = new unsigned int[mNoClasses];
	for (c = 0; c < mNoClasses; c++)
		mpNoSVsOrigSVM[c] = 0;

	for (t = 0; t < mNoExamplesOfOrigSVM; t++)
	{
		bool allreadyCountedSV = false;
		for (c = 0; c < mNoClasses; c++)
		{
			labels[t] = c;
			if (mpSVM->getAlpha(t,c) != 0)
			{
				++mNoNonUniqueSVs;
				++mpNoSVsOrigSVM[c];
				if (!allreadyCountedSV)
				{
					this->mNoUniqueSVs++;
					allreadyCountedSV = true;
				}
			}
		}
	}

	if (verbose)
	{
		cout << "Number of unique SVs: " << this->mNoUniqueSVs << endl;
		cout << "Number of non-unique SVs: " << this->mNoNonUniqueSVs << endl;
		cout << "SVs per class: ";
		for  (c = 0; c < mNoClasses-1; c++){ cout << mpNoSVsOrigSVM[c] << ", "; }
		cout << mpNoSVsOrigSVM[mNoClasses-1] << endl;
	}

	// default initialization of approximation algorithm
	// can be overwritten using the respective function
	mTargetNoVecsForApproximatedSVM             = unsigned(0.1 * mNoNonUniqueSVs);
	mNoGradientDescentIterations                = 100;

	// create SVM for holding the approximation
	mpApproximatedSVM = new MultiClassSVM(pSVM->getKernel(), mNoClasses);

	approximatedVectors.resize(unsigned(0), unsigned(0), false);
	mpApproximatedSVM->SetTrainingData(approximatedVectors);

	mpNoExamplesOfApproximatedSVM = &(mpApproximatedSVM->examples);
	*mpNoExamplesOfApproximatedSVM = 0;

	mpApproximatedSVM->inputDimension = mDimension;
	mbIsApproximatedModelValid = true;

	// auxiliary variable for choosing initial vectors
	mpNoVecsDrawn = new unsigned[mNoClasses];
	mpTargetNoVecsPerClass = new unsigned[mNoClasses];
	for (c = 0; c < mNoClasses; c++) {
		mpNoVecsDrawn[c] = 0;
		mpTargetNoVecsPerClass[c] = 0;
	}

	mbPerformedGradientDescent	=	false;
	mpIndexListOfVecsForSelection = NULL;
}


McSvmApproximation::~McSvmApproximation()
{
	delete mpApproximatedSVM;
	delete[] labels;
	delete[] mpNoSVsOrigSVM;
	delete[] mpNoVecsDrawn;
	delete[] mpTargetNoVecsPerClass;
}

// approximate SVM
float McSvmApproximation::approximate()
{
	// TODO
	// calculate number of vectors per class to be chosen
	determineNoOfVectorsPerClassForApproximation();

	bool bApproximationTargetFulfilled = false;

/*	while (!bApproximationTargetFulfilled)
	{
		this->addVecRprop();

		// determine optimal coefficients
		if (!this->calcOptimalAlphaOfApproximatedSVM())
			return -1;

		// calc new offset
		this->calcOffsetForReducedModel();

		if (*mpNoExamplesOfApproximatedSVM == mTargetNoVecsForApproximatedSVM)
			bApproximationTargetFulfilled = true;

		if (verbose) cout << *mpNoExamplesOfApproximatedSVM << "," << flush;
	}
*/
	return 1;
}

void McSvmApproximation::determineNoOfVectorsPerClassForApproximation()
{
	// there should be at least one SV per class
	for( unsigned c = 0; c < mNoClasses; c++ )
		mpTargetNoVecsPerClass[c] = 1;

	// this only approximately leads to the wished number of vectors. but close enough!
	for( unsigned c = 0; c < mNoClasses; c++ )
		mpTargetNoVecsPerClass[c] += (unsigned)( (float) mTargetNoVecsForApproximatedSVM * fmax( 0.0, (float) mpNoSVsOrigSVM[c]-1 ) / fmax(1.0, (float) mNoNonUniqueSVs - mNoClasses ) );

	if( verbose ) {
		cout << "Target number of vectors per class: ";
		for( unsigned c = 0; c < mNoClasses-1; c++ )
			cout << mpTargetNoVecsPerClass[c] << ", ";
		cout << mpTargetNoVecsPerClass[mNoClasses-1] << endl;
	}

	return;
}
