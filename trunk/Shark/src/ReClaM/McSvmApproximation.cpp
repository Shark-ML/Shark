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

#include <ReClaM/ClassificationError.h>

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
	mNoClasses			     = mpSVM->getClasses();
	mNoExamplesOfOrigSVM = mpSVM->getExamples();   // include those examples with \alpha_i = 0

	//labels = new double[mNoExamplesOfOrigSVM * mNoClasses]; // cheat to get a 2D array

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
			//labels[t * mNoClasses + c] = 0;
			if (mpSVM->getAlpha(t,c) != 0)
			{
			  //labels[t * mNoClasses + c] = 1;
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
	//delete[] labels;
	delete[] mpNoSVsOrigSVM;
	delete[] mpNoVecsDrawn;
	delete[] mpTargetNoVecsPerClass;
}


McSvmApproximationErrorFunction::McSvmApproximationErrorFunction(McSvmApproximation* svmApprox, unsigned int classLabel)
{
	mpApproxSVM = svmApprox;
	mClassLabel = classLabel;
}


// determine difference between original SVM and its approximation
double McSvmApproximation::error()
{
	unsigned int i, j;

	MultiClassSVM &originalSVM     = *mpSVM ;
	MultiClassSVM &approximatedSVM = *mpApproximatedSVM;

	const Array<double>& xxOriginalSVM     = originalSVM.getPoints();
	const Array<double>& xxApproximatedSVM = approximatedSVM.getPoints();

	KernelFunction *KernelOfSVMs = mpKernel;

	double NormOfDiffBetweenSVMs = 0;

	for(unsigned int c = 0; c < mNoClasses; ++c)
	{
    // alpha^2 terms
    for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
      for (j = 0; j < mNoExamplesOfOrigSVM; ++j)
        NormOfDiffBetweenSVMs += originalSVM.getAlpha(i,c) * originalSVM.getAlpha(j,c) * KernelOfSVMs->eval(xxOriginalSVM[i], xxOriginalSVM[j]);

    // beta^2 terms
    for (i = 0; i < *mpNoExamplesOfApproximatedSVM; ++i)
      for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
        NormOfDiffBetweenSVMs += approximatedSVM.getAlpha(i,c) * approximatedSVM.getAlpha(j,c) * KernelOfSVMs->eval(xxApproximatedSVM[i], xxApproximatedSVM[j]);

    // mixed terms
    for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
      for (j = 0; j < *mpNoExamplesOfApproximatedSVM; ++j)
        NormOfDiffBetweenSVMs -= 2 * originalSVM.getAlpha(i,c) * approximatedSVM.getAlpha(j,c) * KernelOfSVMs->eval(xxOriginalSVM[i], xxApproximatedSVM[j]);
	}

	return NormOfDiffBetweenSVMs;
}


double McSvmApproximationErrorFunction::error(Model& model, const Array<double>& input, const Array<double>& target)
{
  MultiClassSVM *mpSVM = mpApproxSVM->mpSVM;
  MultiClassSVM *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;

  const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
  const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();

  KernelFunction *mpKernel = mpSVM->getKernel();
  unsigned int mDimension = mpSVM->getDimension();
  unsigned int mNoExamplesOfOrigSVM = mpSVM->getExamples();
  unsigned int mNoExamplesOfApproximatedSVM = mpApproximatedSVM->getExamples();

  Array<double> additionalVector;
  additionalVector.resize(mDimension, false);
  for(unsigned int i = 0; i < mDimension; ++i)
  {
    additionalVector(i) = model.getParameter(i);
  }

  double functionValue = 0.0;

  for(unsigned int c = 0; c < mpSVM->getClasses(); c++)
  {
    double classFunctionValue = 0.0;

    for(unsigned int i = 0; i < mNoExamplesOfOrigSVM; ++i)
    {
      classFunctionValue += mpSVM->getAlpha(i, c) * mpKernel->eval(xxOriginalSVM[i], additionalVector);
    }

    for(unsigned int i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
    {
      classFunctionValue -= mpApproximatedSVM->getAlpha(i, c) * mpKernel->eval(xxApproximatedSVM[i], additionalVector);
    }

    functionValue += classFunctionValue * classFunctionValue;
  }

  return -functionValue;
}

// determine gradient and actual function value
//double McSvmApproximationErrorFunction::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
//{
//	// TODO verify derivative numerically
//	unsigned int i, j;
//
//	MultiClassSVM  *mpSVM             = mpApproxSVM->mpSVM;
//	MultiClassSVM  *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;
//
//	const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
//	const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();
//
//	KernelFunction *mpKernel               = mpSVM->getKernel();
//	double mGamma                          = mpKernel->getParameter(0);
//	unsigned mDimension                    = mpSVM->getDimension();
//	unsigned mNoExamplesOfOrigSVM          = mpSVM->getExamples();
//	unsigned mNoExamplesOfApproximatedSVM  = mpApproximatedSVM->getExamples();
//
//
//	Array< double > additionalVector;
//  additionalVector.resize(mDimension, false);
//  for (i = 0; i < mDimension ; ++i)
//    additionalVector(i) = model.getParameter(i);
//
//  double value, functionValue = 0;
//  derivative = 0;
//
//  for( unsigned c = 0; c < mpSVM->getClasses(); c++ )
//  {
//
//    for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
//    {
//      value = mpSVM->getAlpha(i,c) * mpKernel->eval(xxOriginalSVM[i], additionalVector);
//      functionValue += value;
//
//      for (j = 0; j < mDimension; ++j)
//        derivative(j) +=  4 * value * mGamma * (xxOriginalSVM(i, j) - additionalVector(j));
//    }
//
//  }
//
//  for( unsigned c = 0; c < mpSVM->getClasses(); c++ )
//  {
//
//    for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
//    {
//      value = mpApproximatedSVM->getAlpha(i,c) * mpKernel->eval(xxApproximatedSVM[i], additionalVector);
//      functionValue -= value;
//
//      for (j = 0; j < mDimension; ++j)
//        derivative(j) -=  4 * value * mGamma * (xxApproximatedSVM(i, j) - additionalVector(j));
//    }
//
//  }
//
//  derivative *= functionValue;
//
//	// make minimization problem and return actual function value
//	derivative *= -1.;
//	return -(functionValue * functionValue);
//}



McSvmApproximationErrorFunctionGlobal::McSvmApproximationErrorFunctionGlobal(McSvmApproximation* svmApprox)
{
	mpApproxSVM = svmApprox;
}

double McSvmApproximationErrorFunctionGlobal::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	return 0.0;
}

// approximate SVM
float McSvmApproximation::approximate(const Array<double> &data, const Array<double> &labels)
{
	// TODO
	// calculate number of vectors per class to be chosen
	determineNoOfVectorsPerClassForApproximation();

	bool bApproximationTargetFulfilled = false;

	while (!bApproximationTargetFulfilled)
	{
//    for(unsigned int i = 0; i < mpApproximatedSVM->getExamples(); ++i)
//      for(unsigned int c = 0; c < mNoClasses; ++c)
//        cout << "(" << c << "," << i << ") " << mpApproximatedSVM->getAlpha(i, c) << endl;
//    cout << endl;

    ZeroOneLoss loss;
    double accuracy = loss.error(*mpApproximatedSVM, data, labels);
    if (verbose) cout << "accuracy (1) " << *mpNoExamplesOfApproximatedSVM << ": " << (1-accuracy)*100.0 << "%," << endl << flush;

    this->addVecRprop(data, labels);

    accuracy = loss.error(*mpApproximatedSVM, data, labels);
    if (verbose) cout << "accuracy (2) " << *mpNoExamplesOfApproximatedSVM << ": " << (1-accuracy)*100.0 << "%," << endl << flush;

    // determine optimal coefficients
		if (!this->calcOptimalAlphaOfApproximatedSVM())
			return -1;

//    for(unsigned int i = 0; i < mpApproximatedSVM->getExamples(); ++i)
//      for(unsigned int c = 0; c < mNoClasses; ++c)
//        cout << "(" << c << "," << i << ") " << mpApproximatedSVM->getAlpha(i, c) << endl;
//    cout << endl;

    accuracy = loss.error(*mpApproximatedSVM, data, labels);
    if (verbose) cout << "accuracy (3) " << *mpNoExamplesOfApproximatedSVM << ": " << (1-accuracy)*100.0 << "%," << endl << flush;

		// calc new offset
		this->calcOffsetForReducedModel();

		//cout << "example 0: " << mpApproximatedSVM->getPoints()[0] << endl;

		if (*mpNoExamplesOfApproximatedSVM == mTargetNoVecsForApproximatedSVM)
			bApproximationTargetFulfilled = true;

    //ZeroOneLoss loss;
    accuracy = loss.error(*mpApproximatedSVM, data, labels);
    if (verbose) cout << "accuracy (4) " << *mpNoExamplesOfApproximatedSVM << ": " << (1-accuracy)*100.0 << "%," << endl << flush;

    if (verbose) cout << *mpNoExamplesOfApproximatedSVM << "," << flush;
	}

	return 1;
}


//double McSvmApproximationErrorFunctionGlobal::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
//{
//	unsigned int k, m, i;
//
//	MultiClassSVM  *mpSVM             = mpApproxSVM->mpSVM;
//	MultiClassSVM  *mpApproximatedSVM = mpApproxSVM->mpApproximatedSVM;
//
//	const Array<double>& xxOriginalSVM     = mpSVM->getPoints();
//	const Array<double>& xxApproximatedSVM = mpApproximatedSVM->getPoints();
//
//	KernelFunction *mpKernel               = mpSVM->getKernel();
//	double mGamma                          = mpKernel->getParameter(0);
//	unsigned mDimension                    = mpSVM->getDimension();
//	unsigned mNoExamplesOfOrigSVM          = mpSVM->getExamples();
//	unsigned mNoExamplesOfApproximatedSVM  = mpApproximatedSVM->getExamples();
//
//	derivative = 0;
//
//	double kernelEval;
//
//	// the coefficients with respect to all elements of the approximating vectors
//	for (k = 0; k < mNoExamplesOfApproximatedSVM; ++k)
//	{
//		for (m = 0; m < mDimension; ++m)
//		{
//			for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
//			{
//				// TODO eliminate kernel evaluations by changing order of loops?
//				kernelEval = mpKernel->eval(xxOriginalSVM[i], xxApproximatedSVM[k]);
//				//derivative(k*mDimension + m) -= 4 * mpSVM->getAlpha(i) * mpApproximatedSVM->getAlpha(k) * (mGamma * kernelEval * (xxOriginalSVM(i, m) - xxApproximatedSVM(k, m)));
//			}
//
//			for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
//			{
//				kernelEval = mpKernel->eval(xxApproximatedSVM[i], xxApproximatedSVM[k]);
//				//derivative(k*mDimension + m) += 4 * mpApproximatedSVM->getAlpha(i) * mpApproximatedSVM->getAlpha(k) * (mGamma * kernelEval * (xxApproximatedSVM(i, m) - xxApproximatedSVM(k, m)));
//			}
//		}
//	}
//
//
//	// the derivatives with respect to the coefficients
//	double sum;
//
//	for (k = 0; k < mNoExamplesOfApproximatedSVM; ++k)
//	{
//		sum = 0;
////		for (i = 0; i < mNoExamplesOfOrigSVM; ++i)
////			sum -= 2 * mpSVM->getAlpha(i) * mpKernel->eval(xxOriginalSVM[i], xxApproximatedSVM[k]) ;
////
////		for (i = 0; i < mNoExamplesOfApproximatedSVM; ++i)
////			sum += 2 * mpApproximatedSVM->getAlpha(i) * mpKernel->eval(xxApproximatedSVM[i], xxApproximatedSVM[k]);
//
//		derivative(mNoExamplesOfApproximatedSVM * mDimension + k) = sum;
//	}
//
//	return mpApproxSVM->error();
//}


// do gradient descent on all parameters of the approximated SVM
float McSvmApproximation::gradientDescent()
{
	unsigned int p;

	Array<double> modelVec;
	//    Array<double> derivativeVec;

	unsigned &noExamples =  *mpNoExamplesOfApproximatedSVM;


	// determine dimension of optimization space and initialize modelVec & derivativeVec;
	unsigned dimensionOfOptimizationSpace = noExamples * mDimension + noExamples * mNoClasses;
	modelVec.resize(dimensionOfOptimizationSpace, false);
	//    derivativeVec.resize(dimensionOfOptimizationSpace, false);

	// write SVM to modelVec, which is input to rPropPlus
	for(unsigned int i = 0; i < noExamples; ++i)
		for(unsigned int j = 0; j < mDimension; ++j)
			modelVec(i*mDimension + j) = approximatedVectors(i, j);

  for(unsigned int i = 0; i < noExamples; ++i)
    for(unsigned int c = 0; c < mNoClasses; ++c)
      modelVec(noExamples * mDimension + i * mNoClasses + c) = mpApproximatedSVM->getAlpha(i,c);


	McSvmApproximationModel model(modelVec);
	McSvmApproximationErrorFunctionGlobal errFunc(this);

	Array<double> DeltaInit;
	DeltaInit.resize(dimensionOfOptimizationSpace, false);
	for(unsigned int i = 0; i < noExamples * mDimension; ++i)
		DeltaInit(i) = 1e-4 * sqrt(1. / (2 * mGamma * mDimension));

	// the initial step size for $\alpha_i$
	for(unsigned int i = 0; i < noExamples * mNoClasses; ++i)
		DeltaInit(noExamples * mDimension + i) = 1e-8;

	IRpropPlus iRpropFunctionMinimizer;
	iRpropFunctionMinimizer.initUserDefined(model, DeltaInit);
	//( model, 1.2, 0.5, 50, 1e-6, 1e-4*sqrt(1./(2.*mGamma))/sqrt(mDimension));

	if (verbose) cout << "GradientDescent " << endl;

	for (p = 1; p <= mNoGradientDescentIterations; ++p)
	{
		Array<double> dummy;
		iRpropFunctionMinimizer.optimize(model, errFunc, dummy, dummy);

		// write modelVec to SVM
		for(unsigned int i = 0; i < noExamples; ++i)
			for(unsigned int j = 0; j < mDimension; ++j)
				approximatedVectors(i, j) = model.getParameter(i * mDimension + j);

    for(unsigned int i = 0; i < noExamples; ++i)
      for(unsigned int c = 0; c < mNoClasses; ++c)
		    mpApproximatedSVM->setAlpha(i, c, model.getParameter(noExamples * mDimension + i * mNoClasses + c));

		if (verbose) cout << p << "," << flush;
	}

	// recompute Thresh
	calcOffsetForReducedModel();

	mbPerformedGradientDescent = true;

	return (float)this->error();
}

// add single vector to SVM approximation
// using iRpropPlus as optimization method
void McSvmApproximation::addVecRprop(const Array<double> &data, const Array<double> &labels)
{
	unsigned int j;

	Array<double> z;
	double error = 1e5;;
	double errorLastIteration = 1e7;
	unsigned counterNoSuccessiveImprovement = 0;

	mbIsApproximatedModelValid = false;

//	double maxAlpha = 0;
//	for(unsigned c = 0; c < mpSVM->getClasses(); c++)
//	  for(unsigned i = 0; i < mpSVM->getExamples(); i++) {
//	    double falpha = fabs(mpSVM->getAlpha(i, c));
//	    maxAlpha = max( falpha, maxAlpha );
//	  }
//	cout << "max. alpha: " << maxAlpha << endl;

	// choose initial vector
	unsigned int class_label = this->chooseVectorForNextIteration(z);

//  maxAlpha = 0;
//  for(unsigned c = 0; c < mpSVM->getClasses(); c++)
//    for(unsigned i = 0; i < mpSVM->getExamples(); i++) {
//      double falpha = fabs(mpSVM->getAlpha(i, c));
//      maxAlpha = max( falpha, maxAlpha );
//    }
//  cout << "max. alpha: " << maxAlpha << endl;
//  cout << endl;

	McSvmApproximationModel model(z);
	McSvmApproximationErrorFunction errFunc(this, class_label);
	errFunc.setEpsilon(1e-6);

  RpropMinus iRpropFunctionMinimizer;
  //IRpropPlus iRpropFunctionMinimizer;
	// TODO check theses parameters! consider number of classes?
	//iRpropFunctionMinimizer.initUserDefined(model, 1.2, 0.5, 1e100, 1e-100, 1e-4*sqrt(1. / (2.*mGamma)) / sqrt((double)mDimension));
  iRpropFunctionMinimizer.init(model);

  //cout << "***" << endl;

  iRpropFunctionMinimizer.mDerrivativeError = 0.0;

	while (counterNoSuccessiveImprovement < 5)
	{
		// do one rProp minimization step
		Array<double> dummy;
		error = iRpropFunctionMinimizer.optimize(model, errFunc, dummy, dummy);

//		cout << "error: " << error << endl;
//    cout << "error diff: " << fabs(errorLastIteration - error) << endl;
//    cout << endl;

//    ZeroOneLoss loss;
//    double accuracy = loss.error(*mpApproximatedSVM, data, labels);
//    if (verbose) cout << "accuracy: " << (1-accuracy)*100.0 << "%" << endl << flush;

		if (fabs((errorLastIteration - error)) < fabs(error * 1e-7))
			++counterNoSuccessiveImprovement;
		else
			counterNoSuccessiveImprovement = 0;

		errorLastIteration = error;
	}

  //cout << "max. derrivative error: " << iRpropFunctionMinimizer.mDerrivativeError << endl;



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

  // the additional mNoClasses is for the thresholds
  mpApproximatedSVM->parameter.resize(mNoClasses * *mpNoExamplesOfApproximatedSVM + mNoClasses, false);

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
			  if(!mpSVM->isSupportVector(j))
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
			double alphaTmp = mpSVM->getAlpha(i,c);
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

	  // TODO remove
	  mpApproximatedSVM->parameter(mNoClasses * *mpNoExamplesOfApproximatedSVM + c) = 0.0;
	  continue;

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
unsigned int McSvmApproximation::chooseVectorForNextIteration(Array<double> &vec)
{
  int label;
	unsigned index_example;
	bool bRepeatDraw = false;

	vec.resize(mDimension, false);

	do
	{
		index_example = Rng::discrete(0, mNoExamplesOfOrigSVM - 1);

		bRepeatDraw = false;
    label = -1;

		// we need a support vector
		if(!mpSVM->isSupportVector(index_example)) {
      bRepeatDraw = true;
      continue;
		}

		while(label < 0) {
		  int index_class = Rng::discrete(0, mNoClasses - 1);
//		  cout << "index_class: " << index_class << endl;
		  if(mpSVM->getAlpha(index_example, index_class) != 0) {
		    label = index_class;
		  }
//      cout << "label: " << label << endl << endl;
		}

		if( mpNoVecsDrawn[label] >= mpTargetNoVecsPerClass[label] ) {
		  bRepeatDraw = true;
		}

//    cout << "*indexSV: " << index_example << endl;
//    cout << "*label: " << label << endl;
//    cout << "*mpNoVecsDrawn[label]: " << mpNoVecsDrawn[label] << endl;
//    cout << "*mpTargetNoVecsPerClass[label]: " << mpTargetNoVecsPerClass[label] << endl;
//    cout << endl;
	}
	while (mpSVM->isSupportVector(index_example) == false || bRepeatDraw == true);

//	cout << "indexSV: " << index_example << endl;
	vec = mpSVM->getPoints()[index_example];

	//unsigned label = labels[indexSV];
	mpNoVecsDrawn[label] += 1;

	return label;
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
