//===========================================================================
/*!
 *  \file SvmApproximation.cpp
 *
 *  \author  T. Suttorp
 *  \date    2006
 *
 *  \brief Approximation of Support Vector Machines (SVMs)
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
#include <ReClaM/SvmApproximation.h>
#include <ReClaM/KernelKMeans.h>


using namespace std;


SvmApproximation::SvmApproximation(SVM *pSVM, bool verbose)
{
	this->verbose = verbose;

	unsigned int t;

	mpSVM             = pSVM;  // store pointer to original SVM
	mpApproximatedSVM = NULL;

	// get parameters of original SVM
	mpKernel             = mpSVM->getKernel();
	mGamma               = mpKernel->getParameter(0);
	mDimension           = mpSVM->getDimension();
	mNoExamplesOfOrigSVM = mpSVM->getExamples();   // include those examples with \alpha_i = 0

	labels = new double[mNoExamplesOfOrigSVM];

	// determine number of SVs of original SVM
	mNoSVs = 0;
	mNoPositiveSVsOfOrigSVM = 0;
	mNoNegativeSVsOfOrigSVM = 0;

	for (t = 0; t < mNoExamplesOfOrigSVM; t++)
	{
		if (mpSVM->getAlpha(t) != 0.0)
		{
			this->mNoSVs++;
			if (mpSVM->getAlpha(t) > 0)
			{
				++mNoPositiveSVsOfOrigSVM;
				labels[t] = 1;
			}
			else
			{
				++mNoNegativeSVsOfOrigSVM;
				labels[t] = -1;
			}
		}
	}

	if (verbose)
	{
		cout << "noVectors: " << 	this->mNoSVs << endl;
		cout << "PositiveSVs, NegativeSVs: "   <<  mNoPositiveSVsOfOrigSVM << ", " << mNoNegativeSVsOfOrigSVM << endl;
	}

	// default initialization of approximation algorithm
	// can be overwritten using the respective function
	mApproximationAlgorithm                     = iRpropPlus;
	mVectorGenerationMode                       = randomSelection;
	mVectorGenerationMode                       = stochasticUniversalSampling;
	if (verbose) cout << mVectorGenerationMode << endl;
	mApproximationTarget                        = classificationRateOnSVs;
	mTargetNoVecsForApproximatedSVM             = unsigned(0.1 * mNoSVs);
	mTargetClassificationRateForApproximatedSVM = 1.0;
	mNoGradientDescentIterations                = 100;

	// create SVM for holding the approximation
	mpApproximatedSVM = new SVM(pSVM->getKernel());

	approximatedVectors.resize(unsigned(0), unsigned(0), false);
	mpApproximatedSVM->SetTrainingData(approximatedVectors);

	mpNoExamplesOfApproximatedSVM = &(mpApproximatedSVM->examples);
	*mpNoExamplesOfApproximatedSVM = 0;

	mpApproximatedSVM->inputDimension = mDimension;
	mbIsApproximatedModelValid = true;


	// auxiliary variable for choosing initial vectors
	mNoPositiveVecsDrawn  = 0;
	mNoNegativeVecsDrawn  = 0;

	mbPerformedGradientDescent	=	false;
	mpIndexListOfVecsForSelection = NULL;
}


SvmApproximation::~SvmApproximation()
{
	delete mpApproximatedSVM;
	delete[] labels;
}



// determine difference between original SVM and its approximation
double SvmApproximation::error()
{
	unsigned int i, j;

	SVM &originalSVM     = *mpSVM ;
	SVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	KernelFunction *KernelOfSVMs = mpKernel;

	double NormOfDiffBetweenSVMs = 0;

	// alpha^2 terms
	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
			NormOfDiffBetweenSVMs += originalSVM.getAlpha(i) * originalSVM.getAlpha(j) * KernelOfSVMs->eval(xxOriginalSVM[i], xxOriginalSVM[j]);

	// beta^2 terms
	for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
			NormOfDiffBetweenSVMs += approximatedSVM.getAlpha(i) * approximatedSVM.getAlpha(j) * KernelOfSVMs->eval(xxApproximatedSVM[i], xxApproximatedSVM[j]);

	// mixed terms
	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
			NormOfDiffBetweenSVMs -= 2 * originalSVM.getAlpha(i) * approximatedSVM.getAlpha(j) * KernelOfSVMs->eval(xxOriginalSVM[i], xxApproximatedSVM[j]);

	return NormOfDiffBetweenSVMs;
}


SvmApproximationErrorFunction::SvmApproximationErrorFunction(SvmApproximation* svmApprox)
{
	mpApproxSVM = svmApprox;
}


double SvmApproximationErrorFunction::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	return 0.0;
}

// determine gradient and actual function value
double SvmApproximationErrorFunction::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	unsigned int i, j;

	SVM  *mpSVM             = mpApproxSVM->mpSVM;
	SVM  *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
	const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();

	KernelFunction *mpKernel               = mpSVM->getKernel();
	double mGamma                          = mpKernel->getParameter(0);
	unsigned mDimension                    = mpSVM->getDimension();
	unsigned mNoExamplesOfOrigSVM          = mpSVM->getExamples();
	unsigned mNoExamplesOfApproximatedSVM  = mpApproximatedSVM->getExamples();


	Array< double > additionalVector;
	additionalVector.resize(mDimension, false);
	for (i = 0; i < mDimension ; ++i)
		additionalVector(i) = model.getParameter(i);


	double value, functionValue = 0;
	derivative = 0;

	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
	{
		value = mpSVM->getAlpha(i) * mpKernel->eval(xxOriginalSVM[i], additionalVector);
		functionValue += value;

		for (j = 0; j < mDimension; ++j)
			derivative(j) +=  4 * value * mGamma * (xxOriginalSVM(i, j) - additionalVector(j));
	}

	for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
	{
		value = mpApproximatedSVM->getAlpha(i) * mpKernel->eval(xxApproximatedSVM[i], additionalVector);
		functionValue -= value ;

		for (j = 0; j < mDimension; ++j)
			derivative(j) -=  4 * value * mGamma * (xxApproximatedSVM(i, j) - additionalVector(j));
	}

	derivative *= functionValue;

	// make minimization problem and return actual function value
	derivative *= -1.;
	return -(functionValue * functionValue);
}



SvmApproximationErrorFunctionGlobal::SvmApproximationErrorFunctionGlobal(SvmApproximation* svmApprox)
{
	mpApproxSVM = svmApprox;
}

double SvmApproximationErrorFunctionGlobal::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	return 0.0;
}

// approximate SVM
float SvmApproximation::approximate()
{
	// calculate number of positive and negative vectors to be chosen
	determineNoPositiveAndNegativeVectorsForApproximation();

	bool bApproximationTargetFulfilled = false;

	while (!bApproximationTargetFulfilled)
	{
		if (mApproximationAlgorithm == iRpropPlus)
			this->addVecRprop();
		else
			if (mApproximationAlgorithm == fixedPointIteration)
				do
				{
					this->addVecFixPointIteration();
				}
				while (!mbIsApproximatedModelValid);

		// determine optimal coefficients
		if (!this->calcOptimalAlphaOfApproximatedSVM())
			return -1;

		// calc new offset
		this->calcOffsetForReducedModel();

		if ((mApproximationTarget == noSVs) && (*mpNoExamplesOfApproximatedSVM == mTargetNoVecsForApproximatedSVM))
			bApproximationTargetFulfilled = true;

		if (mApproximationTarget == classificationRateOnSVs)
		{
			double classificationRate =  this->getClassificationRateOnSVsCorrectlyClassifiedByOrigSVM();

			if (classificationRate >= mTargetClassificationRateForApproximatedSVM)
				bApproximationTargetFulfilled = true;
		}
		if (verbose) cout << *mpNoExamplesOfApproximatedSVM << "," << flush;
	}

	return 1;
}


double SvmApproximationErrorFunctionGlobal::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	unsigned int k, m, i;

	SVM  *mpSVM             = mpApproxSVM->mpSVM;
	SVM  *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
	const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();

	KernelFunction *mpKernel               = mpSVM->getKernel();
	double mGamma                          = mpKernel->getParameter(0);
	unsigned mDimension                    = mpSVM->getDimension();
	unsigned mNoExamplesOfOrigSVM          = mpSVM->getExamples();
	unsigned mNoExamplesOfApproximatedSVM  = mpApproximatedSVM->getExamples();

	derivative = 0;

	double kernelEval;

	// the coefficients with respect to all elements of the approximating vectors
	for (k = 0; k < mNoExamplesOfApproximatedSVM; ++k)
	{
		for (m = 0; m < mDimension; ++m)
		{
			for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
			{
				// TODO eliminate kernel evaluations by changing order of loops?
				kernelEval = mpKernel->eval(xxOriginalSVM[i], xxApproximatedSVM[k]);
				derivative(k*mDimension + m) -= 4 * mpSVM->getAlpha(i) * mpApproximatedSVM->getAlpha(k) * (mGamma * kernelEval * (xxOriginalSVM(i, m) - xxApproximatedSVM(k, m)));
			}

			for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
			{
				kernelEval = mpKernel->eval(xxApproximatedSVM[i], xxApproximatedSVM[k]);
				derivative(k*mDimension + m) += 4 * mpApproximatedSVM->getAlpha(i) * mpApproximatedSVM->getAlpha(k) * (mGamma * kernelEval * (xxApproximatedSVM(i, m) - xxApproximatedSVM(k, m)));
			}
		}
	}


	// the derivatives with respect to the coefficients
	double sum;

	for (k = 0; k < mNoExamplesOfApproximatedSVM; ++k)
	{
		sum = 0;
		for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
			sum -= 2 * mpSVM->getAlpha(i) * mpKernel->eval(xxOriginalSVM[i], xxApproximatedSVM[k]) ;

		for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
			sum += 2 * mpApproximatedSVM->getAlpha(i) * mpKernel->eval(xxApproximatedSVM[i], xxApproximatedSVM[k]);

		derivative(mNoExamplesOfApproximatedSVM * mDimension + k) = sum;
	}

	return mpApproxSVM->error();
}


// do gradient descent on all parameters of the approximated SVM
float SvmApproximation::gradientDescent()
{
	unsigned int i, j, p;

	Array<double> modelVec;
	//    Array<double> derivativeVec;

	unsigned &noExamples =  *mpNoExamplesOfApproximatedSVM;


	// determine dimension of optimization space and initialize modelVec & derivativeVec;
	unsigned dimensionOfOptimizationSpace = noExamples * mDimension  + noExamples;
	modelVec.resize(dimensionOfOptimizationSpace, false);
	//    derivativeVec.resize(dimensionOfOptimizationSpace, false);

	// write SVM to modelVec, which is input to rPropPlus
	for (i = 0; i < noExamples; ++i)
		for (j = 0; j < mDimension; ++j)
			modelVec(i*mDimension + j) = approximatedVectors(i, j);

	for (i = 0; i < noExamples; ++i)
		modelVec(noExamples * mDimension + i) = mpApproximatedSVM->getAlpha(i);


	SvmApproximationModel model(modelVec);
	SvmApproximationErrorFunctionGlobal errFunc(this);

	Array<double> DeltaInit;
	DeltaInit.resize(dimensionOfOptimizationSpace, false);
	for (i = 0; i < noExamples * mDimension; ++i)
		DeltaInit(i) = 1e-4 * sqrt(1. / (2 * mGamma * mDimension));

	// the initial step size for $\alpha_i$
	for (i = 0; i < noExamples; ++i)
		DeltaInit(noExamples * mDimension + i) = 1e-8;

	//    IRpropPlusV iRpropFunctionMinimizer;
	IRpropPlus iRpropFunctionMinimizer;
	iRpropFunctionMinimizer.initUserDefined(model, DeltaInit);
	//( model, 1.2, 0.5, 50, 1e-6, 1e-4*sqrt(1./(2.*mGamma))/sqrt(mDimension));

	if (verbose) cout << "GradientDescent " << endl;

	for (p = 1; p <= mNoGradientDescentIterations; ++p)
	{
		Array<double> dummy;
		iRpropFunctionMinimizer.optimize(model, errFunc, dummy, dummy);

		// write modelVec to SVM
		for (i = 0; i < noExamples; ++i)
			for (j = 0; j < mDimension; ++j)
				approximatedVectors(i, j) = model.getParameter(i * mDimension + j);

		for (i = 0; i < noExamples; ++i)
			mpApproximatedSVM->parameter(i) =  model.getParameter(noExamples * mDimension + i);

		if (verbose) cout << p << "," << flush;
	}

	// recompute Thresh
	calcOffsetForReducedModel();

	mbPerformedGradientDescent = true;

	return (float)this->error();
}

// add single vector to SVM approximation
// using iRpropPlus as optimization method
void SvmApproximation::addVecRprop()
{
	unsigned int j;

	Array<double> z;
	double error = 1e5;;
	double errorLastIteration = 1e7;
	unsigned counterNoSuccessiveImprovement = 0;

	mbIsApproximatedModelValid = false;

	// choose initial vector
	this->chooseVectorForNextIteration(z);

	SvmApproximationModel model(z);
	SvmApproximationErrorFunction errFunc(this);

	IRpropPlus iRpropFunctionMinimizer;
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

// add single vector to SVM approximation
// using fixpoint iteration
void SvmApproximation::addVecFixPointIteration()
{
	unsigned int i, j;

	SVM &originalSVM     = *mpSVM ;
	SVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	mbIsApproximatedModelValid = false;


	Array<double> z, zOld;
	double tmp, functionValue;;
	unsigned counter = 50;


	// choose initial vector
	this->chooseVectorForNextIteration(z);


	while (--counter > 0)  // iteration steps of fixed-point iteration
	{
		functionValue = 0;
		zOld = z;
		z = 0;

		for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		{
			tmp = mpSVM->getAlpha(i) * mpKernel->eval(xxOriginalSVM[i], zOld);
			functionValue += tmp ;

			for (j = 0; j < mDimension; ++j)
				z(j) += tmp * xxOriginalSVM(i, j);
		}

		for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
		{
			tmp = mpApproximatedSVM->getAlpha(i) * mpKernel->eval(xxApproximatedSVM[i], zOld);
			functionValue -= tmp ;

			for (j = 0; j < mDimension; ++j)
				z(j) -= tmp * xxApproximatedSVM(i, j);
		}

		// discard actual initial vector and choose new one
		if (fabs(functionValue) < 1e-7)
		{
			if (verbose) cout << "." << flush;

			//rearrange elems in list
			if (mpIndexListOfVecsForSelection != NULL)  // vectorGenerationMode is stochasticUniversalSampling or kMeansClustering
				std::random_shuffle(mpIndexListOfVecsForSelection[0].begin() + *mpNoExamplesOfApproximatedSVM, mpIndexListOfVecsForSelection->end());

			if (labels[mIndexOfChosenVec] > 0)
				--mNoPositiveVecsDrawn;
			else
				--mNoNegativeVecsDrawn;

			return;
		}

		z /= functionValue;
	}

	(*mpNoExamplesOfApproximatedSVM)++;

	// add vector to existing approximation of SVM
	approximatedVectors.resize(*mpNoExamplesOfApproximatedSVM, mDimension, true);
	for (j = 0; j < mDimension; ++j)
		approximatedVectors(*mpNoExamplesOfApproximatedSVM - 1, j) = z(j);

	mbIsApproximatedModelValid = true;
}


// determine optimal coefficients for SVM approximation
bool SvmApproximation::calcOptimalAlphaOfApproximatedSVM()
{
	unsigned int i, j, k;

	SVM &originalSVM     = *mpSVM ;
	SVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

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
	Kzx.resize(*mpNoExamplesOfApproximatedSVM, mNoSVs, false);

	unsigned SVcounterForOrigSVM;

	for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
	{
		SVcounterForOrigSVM = 0;
		for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
		{
			if (mpSVM->getAlpha(j) == 0)
				continue;

			Kzx(i, SVcounterForOrigSVM++) =  mpKernel->eval(xxApproximatedSVM[i], xxOriginalSVM[j]);
		}
	}

	Array<double> KzInvDotKzx;
	KzInvDotKzx = innerProduct(KzInv, Kzx);

	Array<double> AlphaY;
	AlphaY.resize(mNoSVs, false);

	SVcounterForOrigSVM = 0;

	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
	{
		double alphaTmp =  mpSVM->getAlpha(i);
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
		for (k = 0; k < mNoSVs; ++k)
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
			mpApproximatedSVM->setParameter(i, Beta(i));
	}
	return true;
}

// determine "optimal" offset for SVM approximation
bool SvmApproximation::calcOffsetForReducedModel()
{
	unsigned int i, j;

	SVM &originalSVM     = *mpSVM ;
	SVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	double sum = 0;
	double b = mpSVM->getOffset();

	// calculate the mean over all differences between the original and the reduced SVM
	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
	{
		if (mpSVM->getAlpha(i) == 0)
			continue;

		for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
			sum += mpSVM->getAlpha(j) * mpKernel->eval(xxOriginalSVM[j], xxOriginalSVM[i]);

		sum += b;

		for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
			sum -= mpApproximatedSVM->getAlpha(j) * mpKernel->eval(xxApproximatedSVM[j],  xxOriginalSVM[i]);
	}

	mOffsetOfApproximatedSVM = sum / (double)this->mNoSVs;

	// set offset of approximated SVM
	mpApproximatedSVM->parameter(*mpNoExamplesOfApproximatedSVM) = mOffsetOfApproximatedSVM ;

	return true;
}


void SvmApproximation::createIndexListWithKMeans()
{
	unsigned int i, q;

	SVM &originalSVM = *mpSVM ;

	std::vector<int> indexListOfPositiveSVs;
	std::vector<int> indexListOfNegativeSVs;

	if (mpIndexListOfVecsForSelection == NULL)
		mpIndexListOfVecsForSelection = new(std::vector<int>);

	mpIndexListOfVecsForSelection->reserve(mTargetNoVecsForApproximatedSVM);


	indexListOfPositiveSVs.clear();
	indexListOfNegativeSVs.clear();

	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
	{
		if (originalSVM.getAlpha(i) == 0)
			continue;

		if (labels[i] > 0)
			indexListOfPositiveSVs.push_back(i);
		else
			indexListOfNegativeSVs.push_back(i);
	}


	KernelKMeans kernelClustering1(mpKernel, mpSVM->getPoints(), indexListOfPositiveSVs, mTargetNoPositiveVectors);
	std::vector<int> *pIndexList = kernelClustering1.clusterVectors();
	for (q = 0; q < pIndexList->size(); ++q)
		(*mpIndexListOfVecsForSelection).push_back((*pIndexList)[q]);
	delete pIndexList;


	KernelKMeans kernelClustering2(mpKernel, mpSVM->getPoints(), indexListOfNegativeSVs, mTargetNoNegativeVectors);
	pIndexList = kernelClustering2.clusterVectors();
	for (q = 0; q < pIndexList->size(); ++q)
		(*mpIndexListOfVecsForSelection).push_back((*pIndexList)[q]);
	delete pIndexList;

	std::random_shuffle(mpIndexListOfVecsForSelection[0].begin(), mpIndexListOfVecsForSelection->end());
}


void SvmApproximation::createIndexListWithStochasticUniversalSampling()
{
	unsigned int i;

	SVM &originalSVM = *mpSVM ;
	double alphaSum = 0;

	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		alphaSum += fabs(originalSVM.getAlpha(i));

	// create permutation vector
	std::vector<int> permVec(mNoExamplesOfOrigSVM);
	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
		permVec[i] = i;

	std::random_shuffle(permVec.begin(), permVec.end());

	float frac = (float)(alphaSum / (double)mTargetNoVecsForApproximatedSVM);
	Uniform abc(0.0, frac);
	double p = abc();

	if (mpIndexListOfVecsForSelection == NULL)
		mpIndexListOfVecsForSelection	= new(std::vector<int>);

	mpIndexListOfVecsForSelection->reserve(mTargetNoVecsForApproximatedSVM);


	double sum = 0;
	unsigned selectedElems = 0;
	i = 0;

	while (i < mNoExamplesOfOrigSVM)
	{
		sum +=  fabs(originalSVM.getAlpha(permVec[i]));

		while (sum >= p + selectedElems * frac)
		{
			selectedElems++;
			mpIndexListOfVecsForSelection->push_back(permVec[i]);
		}
		i++;
	}
}



// choose SV from original SVM
bool SvmApproximation::chooseVectorForNextIteration(Array<double> &vec)
{
// 	unsigned i;
	unsigned indexSV;
	bool bRepeatDraw = false;

	vec.resize(mDimension, false);

	switch (mVectorGenerationMode)
	{
	case randomSelection:
	{
		do
		{
			indexSV = Rng::discrete(0, mNoExamplesOfOrigSVM - 1);

			bRepeatDraw = false;

			if (labels[indexSV] > 0 && mNoPositiveVecsDrawn >= mTargetNoPositiveVectors)
				bRepeatDraw = true;

			if (labels[indexSV] < 0 && mNoNegativeVecsDrawn >= mTargetNoNegativeVectors)
				bRepeatDraw = true;

		}
		while (mpSVM->getAlpha(indexSV) == 0 || bRepeatDraw == true);


		mIndexOfChosenVec = indexSV;

// 		for (i = 0; i < mDimension; ++i)
// 			vec[i] = (mpSVM->getPoints().elemvec() + indexSV * mDimension)[i];
		vec = mpSVM->getPoints()[indexSV];

		if (labels[indexSV] > 0)
			++mNoPositiveVecsDrawn;
		else
			++mNoNegativeVecsDrawn;

		break;
	}

	case kMeansClustering:
	{
		// create index list if this function is called for the first time
		if (mpIndexListOfVecsForSelection == NULL)
			this->createIndexListWithKMeans();


		// elements are randomly shuffled and sequentially used
		unsigned elem = *mpNoExamplesOfApproximatedSVM;
		mIndexOfChosenVec = (*mpIndexListOfVecsForSelection)[elem];

// 		for (i = 0; i < mDimension; ++i)
// 			vec[i] = (mpSVM->getPoints().elemvec() + mIndexOfChosenVec * mDimension)[i];
		vec = mpSVM->getPoints()[mIndexOfChosenVec];

		break;
	}

	case stochasticUniversalSampling:
	{
		// create index list if this function is called for the first time
		if (mpIndexListOfVecsForSelection == NULL)
			this->createIndexListWithStochasticUniversalSampling();

		unsigned elem = *mpNoExamplesOfApproximatedSVM;
		mIndexOfChosenVec = (*mpIndexListOfVecsForSelection)[elem];

// 		for (i = 0; i < mDimension; ++i)
// 			vec[i] = (mpSVM->getPoints().elemvec() + mIndexOfChosenVec * mDimension)[i];
		vec = mpSVM->getPoints()[mIndexOfChosenVec];

		break;
	}

	default:
		throw SHARKEXCEPTION("Undefined VectorGenerationMode");
	}

	return true;
}


// classify all original SVs with the approximated SVM
float SvmApproximation::getClassificationRateOnSVsCorrectlyClassifiedByOrigSVM()
{
	unsigned int i;

	Array<double> result(1, 1);

	unsigned correctlyClassified = 0;
	unsigned incorrectlyClassified = 0;
	unsigned noCorrectlyClassifiedByOriginalSVM = 0;

	for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
	{
		if (mpSVM->getAlpha(i) != 0)
		{
			// prediction of original SVM
			mpSVM->model(mpSVM->getPoints().row(i), result);

			// SV not correctly classified by original SVM
			if (result(0)*labels[i] < 0)
				continue;

			++noCorrectlyClassifiedByOriginalSVM;
			mpApproximatedSVM->model(mpSVM->getPoints().row(i), result);

			if (result(0)*labels[i] > 0)
				++correctlyClassified;
			else
				++incorrectlyClassified;
		}
	}

	return (float)correctlyClassified / (float)noCorrectlyClassifiedByOriginalSVM;
}


void SvmApproximation::determineNoPositiveAndNegativeVectorsForApproximation()
{
	//	use fraction of positive and negative examples of orig. SVs
	//  for choosing initial vectors
	mTargetNoPositiveVectors	= (unsigned)((float)mNoPositiveSVsOfOrigSVM / (float)mNoSVs * (float)mTargetNoVecsForApproximatedSVM) ;

	if (mTargetNoPositiveVectors <= 0)
		mTargetNoPositiveVectors = 1;

	if (mTargetNoPositiveVectors >= mTargetNoVecsForApproximatedSVM)
		mTargetNoPositiveVectors = mTargetNoVecsForApproximatedSVM - 1;

	mTargetNoNegativeVectors =	mTargetNoVecsForApproximatedSVM - mTargetNoPositiveVectors;
}


