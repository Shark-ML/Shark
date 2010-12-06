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


McSvmApproximationErrorFunction::McSvmApproximationErrorFunction(McSvmApproximation* svmApprox)
{
	mpApproxSVM = svmApprox;
}


double McSvmApproximationErrorFunction::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	return 0.0;
}

// determine gradient and actual function value
double McSvmApproximationErrorFunction::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// TODO verify derivative numerically
	unsigned int i, j;

	MultiClassSVM  *mpSVM             = mpApproxSVM->mpSVM;
	MultiClassSVM  *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
	const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();

	KernelFunction *mpKernel               = mpSVM->getKernel();
	double mGamma                          = mpKernel->getParameter(0);
	unsigned mDimension                    = mpSVM->getDimension();
	unsigned mNoExamplesOfOrigSVM          = mpSVM->getExamples();
	unsigned mNoExamplesOfApproximatedSVM  = mpApproximatedSVM->getExamples();

	double functionValue = 0.0;

	for( unsigned c = 0; c < mpSVM->getClasses(); c++ ) {

		Array< double > additionalVector;
		additionalVector.resize(mDimension, false);
		for (i = 0; i < mDimension ; ++i)
			additionalVector(i) = model.getParameter(i);

		double value, tmpFunctionValue = 0;

		Array< double > tmpDerivative;
		tmpDerivative.resize(mDimension, false);
		tmpDerivative = 0;

		for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		{
			value = mpSVM->getAlpha(i,c) * mpKernel->eval(xxOriginalSVM[i], additionalVector);
			tmpFunctionValue += value;

			for (j = 0; j < mDimension; ++j)
				tmpDerivative(j) +=  4 * value * mGamma * (xxOriginalSVM(i, j) - additionalVector(j));
		}

		for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
		{
			value = mpApproximatedSVM->getAlpha(i,c) * mpKernel->eval(xxApproximatedSVM[i], additionalVector);
			tmpFunctionValue -= value ;

			for (j = 0; j < mDimension; ++j)
				tmpDerivative(j) -=  4 * value * mGamma * (xxApproximatedSVM(i, j) - additionalVector(j));
		}

		tmpDerivative *= tmpFunctionValue;
		derivative    += tmpDerivative;
		functionValue += tmpFunctionValue;
	}

	// make minimization problem and return actual function value
	derivative *= -1.;
	return -(functionValue * functionValue);
}



// approximate SVM
float McSvmApproximation::approximate()
{
	// TODO
	// calculate number of vectors per class to be chosen
	determineNoOfVectorsPerClassForApproximation();

	bool bApproximationTargetFulfilled = false;

	while (!bApproximationTargetFulfilled)
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

	return 1;
}


// add single vector to SVM approximation
// using iRpropPlus as optimization method
void McSvmApproximation::addVecRprop()
{
	unsigned int j;

	Array<double> z;
	double error = 1e5;;
	double errorLastIteration = 1e7;
	unsigned counterNoSuccessiveImprovement = 0;

	mbIsApproximatedModelValid = false;

	// choose initial vector
	this->chooseVectorForNextIteration(z);

	McSvmApproximationModel model(z);
	McSvmApproximationErrorFunction errFunc(this);

	IRpropPlus iRpropFunctionMinimizer;
	// TODO check theses parameters! consider number of classes?
	iRpropFunctionMinimizer.initUserDefined(model, 1.2, 0.5, 50, 1e-6, 1e-4*sqrt(1. / (2.*mGamma)) / sqrt((double)mDimension));


	while (counterNoSuccessiveImprovement < 5)
	{
		// do one rProp minimization step
		Array<double> dummy;
		error = iRpropFunctionMinimizer.optimize(model, errFunc, dummy, dummy);

		if (fabs((errorLastIteration - error)) < 1e-7)
			++counterNoSuccessiveImprovement;
		else
			counterNoSuccessiveImprovement = 0;

		errorLastIteration = error;
	}


	// add vector to existing approximation of SVM
	(*mpNoExamplesOfApproximatedSVM)++;
	approximatedVectors.resize(*mpNoExamplesOfApproximatedSVM, mDimension, true);
	for (j = 0; j < mDimension; ++j)
		approximatedVectors(*mpNoExamplesOfApproximatedSVM - 1, j) = model.getParameter(j);
}


// determine optimal coefficients for SVM approximation
bool McSvmApproximation::calcOptimalAlphaOfApproximatedSVM()
{
	unsigned int i, j, k;

	MultiClassSVM &originalSVM     = *mpSVM ;
	MultiClassSVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	for( unsigned c = 0; c < mNoClasses; c++ ) {

		Array2D<double> Kz, KzInv;
		Array2D<double> Kzx;

		double tmp;

		// calculate Kz
		Kz.resize(*mpNoExamplesOfApproximatedSVM, *mpNoExamplesOfApproximatedSVM, false);
		KzInv.resize(*mpNoExamplesOfApproximatedSVM, *mpNoExamplesOfApproximatedSVM, false);

		for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		{
			//($Kz = Kz^T$)
			for (j = i; j < *mpNoExamplesOfApproximatedSVM; ++j)
			{
				tmp = mpKernel->eval(xxApproximatedSVM[i], xxApproximatedSVM[j]);
				Kz(i, j) = tmp;
				Kz(j, i) = tmp;
			}
		}


		try
		{
			g_inverse(Kz, KzInv);
		}
		catch (unsigned int e)
		{
			if (verbose) cout << "EXCEPTION: " << e << endl;
			return false;
		}


		// calc Kzx
		unsigned numberOfSVs = mpSVM->getNumberOfSupportVectors();
		Kzx.resize(*mpNoExamplesOfApproximatedSVM, numberOfSVs, false);

		unsigned SVcounterForOrigSVM;

		for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		{
			SVcounterForOrigSVM = 0;
			for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
			{
				if (mpSVM->getAlpha(j,c) == 0)
					continue;

				Kzx(i, SVcounterForOrigSVM++) =  mpKernel->eval(xxApproximatedSVM[i], xxOriginalSVM[j]);
			}
		}

		Array<double> KzInvDotKzx;
		KzInvDotKzx = innerProduct(KzInv, Kzx);

		Array<double> AlphaY;
		AlphaY.resize(numberOfSVs, false);

		SVcounterForOrigSVM = 0;

		for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		{
			double alphaTmp =  mpSVM->getAlpha(i,c);
			if (alphaTmp == 0)
				continue;
			else
				AlphaY(SVcounterForOrigSVM++) =  alphaTmp;
		}

		// calculate beta
		Array<double> Beta;
		Beta.resize(*mpNoExamplesOfApproximatedSVM, false);

		for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		{
			Beta(i) = 0;
			for (k = 0; k < numberOfSVs; ++k)
				Beta(i) += KzInvDotKzx(i, k) * AlphaY(k);
		}

		// the additional one is for the threshold
		mpApproximatedSVM->parameter.resize(*mpNoExamplesOfApproximatedSVM + 1, false);


		for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		{
			if (isnan(Beta(i)))
			{
				if (verbose) cout << " ... numerical problem; maybe kernel not appropriate for dataset !!!" << endl;
				return false;
			}
			else
				mpApproximatedSVM->setAlpha(i, c, Beta(i));
		}

	}

	return true;
}


// determine "optimal" offset for SVM approximation
bool McSvmApproximation::calcOffsetForReducedModel()
{
	unsigned int i, j;

	MultiClassSVM &originalSVM     = *mpSVM ;
	MultiClassSVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	for(unsigned c = 0; c < mNoClasses; c++) {

    double sum = 0;
    double b = mpSVM->getOffset(c);

    // calculate the mean over all differences between the original and the reduced SVM
    for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
    {
      if (mpSVM->getAlpha(i,c) == 0)
        continue;

      // TODO some kernel evaluations could be dropped
      for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
        sum += mpSVM->getAlpha(j,c) * mpKernel->eval(xxOriginalSVM[j], xxOriginalSVM[i]);

      sum += b;

      for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
        sum -= mpApproximatedSVM->getAlpha(j,c) * mpKernel->eval(xxApproximatedSVM[j],  xxOriginalSVM[i]);
    }

    mOffsetOfApproximatedSVM = (double)sum / (double)mpSVM->getNumberOfSupportVectors(c);

    // set offset of approximated SVM
    mpApproximatedSVM->parameter(mNoClasses * *mpNoExamplesOfApproximatedSVM + c) = mOffsetOfApproximatedSVM ;
	}

	return true;
}


// choose SV from original SVM
bool McSvmApproximation::chooseVectorForNextIteration(Array<double> &vec)
{
	unsigned indexSV;
	bool bRepeatDraw = false;

	vec.resize(mDimension, false);

	do
	{
		indexSV = Rng::discrete(0, mNoExamplesOfOrigSVM - 1);

		bRepeatDraw = false;
		unsigned label = labels[indexSV];

		if( mpNoVecsDrawn[label] >= mpTargetNoVecsPerClass[label] )
			bRepeatDraw = true;
	}
	while (mpSVM->isSupportVector(indexSV) == false || bRepeatDraw == true);

	vec = mpSVM->getPoints()[indexSV];

	unsigned label = labels[indexSV];
	mpNoVecsDrawn[label] += 1;

	return true;
}


void McSvmApproximation::determineNoOfVectorsPerClassForApproximation()
{
	// there should be at least one SV per class
	for( unsigned c = 0; c < mNoClasses; c++ )
		mpTargetNoVecsPerClass[c] = 1;

	// this only approximately leads to the wished number of vectors. but close enough!
	for( unsigned c = 0; c < mNoClasses; c++ )
		mpTargetNoVecsPerClass[c] += (unsigned)( (double)( mTargetNoVecsForApproximatedSVM * std::max<unsigned>( 0, mpNoSVsOrigSVM[c]-1 ) ) / (double) std::max<unsigned>(1, mNoNonUniqueSVs - mNoClasses ) );

	if( verbose ) {
		cout << "Target number of vectors per class: ";
		for( unsigned c = 0; c < mNoClasses-1; c++ )
			cout << mpTargetNoVecsPerClass[c] << ", ";
		cout << mpTargetNoVecsPerClass[mNoClasses-1] << endl;
	}

	return;
}
