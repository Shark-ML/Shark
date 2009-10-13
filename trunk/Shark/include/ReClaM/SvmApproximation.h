//===========================================================================
/*!
 *  \file SvmApproximation.h
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


#ifndef _SvmApproximation_H_
#define _SvmApproximation_H_


#include <Array/Array.h>
#include <ReClaM/Svm.h>


//!
//! \brief Approximation of Support Vector Machines (SVMs)
//!
//! \author T. Suttorp
//! \date 2006
//!
//! \par The SvmApproximator class approximates the decision
//! function of an Shark-SVM object
//!
class SvmApproximation
{
public:
	enum approximationAlgorithm
	{
		iRpropPlus,
		fixedPointIteration
	};

	enum approximationTarget
	{
		noSVs,
		classificationRateOnSVs
	};

	enum vectorGenerationMode
	{
		randomSelection,
		stochasticUniversalSampling,
		kMeansClustering
	};


	//! constructor
	//! \param  pSVM     SVM to be approximated
	SvmApproximation(SVM *pSVM, bool verbose = false);

	//! destructor
	~SvmApproximation();


	//!  approximate SVM
	float approximate();

	//! perform gradient descent on overall model of approximated SVM  (SVs and coefficients)
	float gradientDescent();

	//! return pointer to approximated SVM
	SVM* getApproximatedSVM(){ return mpApproximatedSVM; }
	; //< get approximated SVM



	//!  specify the algorithm for the approximation
	void setApproximationAlgorithm(approximationAlgorithm aA)
	{ mApproximationAlgorithm = aA; };

	//!  specify technique for choosing initial vectors
	void setVectorGenerationMode(int mode){ mVectorGenerationMode = mode; };

	//!  specify the goal
	//!  iterate until the approximated SV consists of a specified number of SVs
	//!  or until a certain classification on the SVs of the original SVM is achieved
	void setApproximationTarget(approximationTarget aT){mApproximationTarget = aT;};

	//! specify the number of the SVs for the approximated SVM
	//! only used if "approximationTarget" is set to "noSVs"
	void setTargetNoVecsForApproximatedSVM(unsigned noVecs)
	{ mTargetNoVecsForApproximatedSVM = noVecs; };

	//! specify the classification rate that should be achieved on the SVs of the original SVM
	//! only used if "approximationTarget" is set to "classificationRateOnSVs"
	void setTargetClassificationRateOnSVsForApproximatedSVM(float classificationRate)
	{ mTargetClassificationRateForApproximatedSVM = classificationRate; };

	void setNoGradientDescentIterations(unsigned no){ mNoGradientDescentIterations = no; }


protected:
	//! in verbose mode the algorithm outputs some helpful comments to stdout.
	bool verbose;

	//! determine "difference" between original and approximated SVM
	double error();

	//! optimize vector using iRpropPlus
	void addVecRprop();

	//! optimize using fixed-point iteration
	void addVecFixPointIteration();

	//! calculate optimal coefficients for approx. SVM
	bool calcOptimalAlphaOfApproximatedSVM();

	//! calculate bias "b" for approx. SVM
	bool calcOffsetForReducedModel();

	//! determine classification rate of approx. SVM on SVs of original SVM
	float getClassificationRateOnSVsCorrectlyClassifiedByOrigSVM();

	void determineNoPositiveAndNegativeVectorsForApproximation();

	//! choose vector from original SVs
	bool chooseVectorForNextIteration(Array<double> &vec);

	//! choose initials vectors with k-means-clustering
	void createIndexListWithStochasticUniversalSampling();

	void createIndexListWithKMeans();



	SVM* mpSVM;              //< pointer to original SVM
	SVM* mpApproximatedSVM;  //< pointer to approximated SVM

	double* labels;

	Array<double> approximatedVectors;

	unsigned	mNoExamplesOfOrigSVM, mDimension, mNoSVs;
	unsigned*   mpNoExamplesOfApproximatedSVM;
	double		mOffsetOfApproximatedSVM;

	// parameters of the approximation algorithm
	unsigned mTargetNoVecsForApproximatedSVM, mTargetNoPositiveVectors, mTargetNoNegativeVectors ;
	float    mTargetClassificationRateForApproximatedSVM;

	approximationAlgorithm mApproximationAlgorithm;
	approximationTarget    mApproximationTarget;

	KernelFunction* mpKernel;
	double          mGamma;

	unsigned int	mVectorGenerationMode ;
	bool		mbIsApproximatedModelValid;
	bool		mbPerformedGradientDescent;

	unsigned        mNoGradientDescentIterations;


	unsigned int	mNoPositiveSVsOfOrigSVM, mNoNegativeSVsOfOrigSVM;
	unsigned int	mNoPositiveVecsDrawn, mNoNegativeVecsDrawn;


	const char* mOutputDir;

	unsigned mIndexOfChosenVec;

	std::vector<int>* mpIndexListOfVecsForSelection;

	friend class SvmApproximationErrorFunction;
	friend class SvmApproximationErrorFunctionGlobal;
};


//! \brief Approximation of Support Vector Machines (SVMs)
class SvmApproximationModel : public Model
{
public:
	SvmApproximationModel(Array<double>& initVec){parameter = initVec;};
	virtual void model(const Array<double>& input, Array<double>& output){};
};


//! \brief Approximation of Support Vector Machines (SVMs)
class SvmApproximationErrorFunction : public ErrorFunction
{
public:
	SvmApproximationErrorFunction(SvmApproximation* svmApprox);

	double error(Model& model, const Array<double>& input, const Array<double>& target);
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	SvmApproximation* mpApproxSVM;
};


//! \brief Approximation of Support Vector Machines (SVMs)
class SvmApproximationErrorFunctionGlobal : public ErrorFunction
{
public:
	SvmApproximationErrorFunctionGlobal(SvmApproximation* svmApprox);

	double error(Model& model, const Array<double>& input, const Array<double>& target);
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	SvmApproximation *mpApproxSVM;
};


#endif

