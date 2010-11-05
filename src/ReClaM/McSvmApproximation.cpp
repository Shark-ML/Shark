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
	mNoSVs = 0;
	mNoSVsOrigSVM = new unsigned int[mNoClasses];
	for (c = 0; c < mNoClasses; c++)
		mNoSVsOrigSVM[c] = 0;

	for (t = 0; t < mNoExamplesOfOrigSVM; t++)
	{
		bool allreadyCountedSV = false;
		for (c = 0; c < mNoClasses; c++)
		{
			labels[t] = c;
			if (mpSVM->getAlpha(t,c) > 0)
			{
				++mNoSVsOrigSVM[c];
				if (!allreadyCountedSV)
				{
					this->mNoSVs++;
					allreadyCountedSV = true;
				}
				// TODO also initialize targets?
			}
		}
	}

	if (verbose)
	{
		cout << "noVectors: " << 	this->mNoSVs << endl;
		cout << "SVs per class: ";
		for  (c = 0; c < mNoClasses-1; c++){ cout << mNoSVsOrigSVM[c] << ", "; }
		cout << mNoSVsOrigSVM[mNoClasses-1] << endl;
	}

	// default initialization of approximation algorithm
	// can be overwritten using the respective function
	mTargetNoVecsForApproximatedSVM             = unsigned(0.1 * mNoSVs);
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
	mNoVecsDrawn = new unsigned int[mNoClasses];
	for (c = 0; c < mNoClasses; c++)
		mNoVecsDrawn[c] = 0;

	mbPerformedGradientDescent	=	false;
	mpIndexListOfVecsForSelection = NULL;
}


McSvmApproximation::~McSvmApproximation()
{
	delete mpApproximatedSVM;
	delete[] labels;
}
