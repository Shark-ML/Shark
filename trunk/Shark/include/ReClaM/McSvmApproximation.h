//===========================================================================
/*!
 *  \file McSvmApproximation.h
 *
 *  \author  B. Weghenkel
 *  \date    2010
 *
 *  \brief Approximation of Multi-class Support Vector Machines (SVMs)
 *
 *  \par Copyright (c) 2010:
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


#ifndef _McSvmApproximation_H_
#define _McSvmApproximation_H_


#include <Array/Array.h>
#include <ReClaM/Svm.h>


//!
//! \brief Approximation of Multi-class Support Vector Machines (SVMs)
//!
//! \author B. Weghenkel
//! \date 2010
//!
//! \par The McSvmApproximator class approximates the decision
//! function of an Multi-class Shark-SVM object
//!
class McSvmApproximation
{
public:

	//! constructor
	//! \param  pSVM     SVM to be approximated
	McSvmApproximation(MultiClassSVM *pSVM, bool verbose = false);

	//! destructor
	~McSvmApproximation();


	//!  approximate SVM
	float approximate();

	//! perform gradient descent on overall model of approximated SVM  (SVs and coefficients)
	float gradientDescent();

	//! return pointer to approximated SVM
	MultiClassSVM* getApproximatedSVM(){ return mpApproximatedSVM; }
	; //< get approximated SVM

	//! specify the number of the SVs for the approximated SVM
	//! only used if "approximationTarget" is set to "noSVs"
	void setTargetNoVecsForApproximatedSVM(unsigned noVecs)
	{ mTargetNoVecsForApproximatedSVM = noVecs; };

	void setNoGradientDescentIterations(unsigned no){ mNoGradientDescentIterations = no; }


protected:
	//! in verbose mode the algorithm outputs some helpful comments to stdout.
	bool verbose;

	//! determine "difference" between original and approximated SVM
	double error();

	//! optimize vector using iRpropPlus
	void addVecRprop();

	//! calculate optimal coefficients for approx. SVM
	bool calcOptimalAlphaOfApproximatedSVM();

	//! calculate bias "b" for approx. SVM
	bool calcOffsetForReducedModel();

	//! determine classification rate of approx. SVM on SVs of original SVM
	float getClassificationRateOnSVsCorrectlyClassifiedByOrigSVM();

	void determineNoOfVectorsPerClassForApproximation();

	//! choose vector from original SVs
	bool chooseVectorForNextIteration(Array<double> &vec);


	MultiClassSVM* mpSVM;              //< pointer to original SVM
	MultiClassSVM* mpApproximatedSVM;  //< pointer to approximated SVM

	double* labels;

	Array<double> approximatedVectors;

	//! Number of examples used as SV somewhere (in one of the machines)
	unsigned	mNoUniqueSVs;
	//! Number of SVs taken together all the different machines
	unsigned	mNoNonUniqueSVs;
	unsigned	mNoExamplesOfOrigSVM, mDimension, mNoClasses;
	unsigned *mpNoExamplesOfApproximatedSVM;
	double		mOffsetOfApproximatedSVM;

	// parameters of the approximation algorithm
	unsigned 	mTargetNoVecsForApproximatedSVM;
	unsigned 	*mpTargetNoVecsPerClass;

	KernelFunction* mpKernel;
	double          mGamma;

	bool			mbIsApproximatedModelValid;
	bool			mbPerformedGradientDescent;

	unsigned        mNoGradientDescentIterations;
	unsigned int  	*mpNoSVsOrigSVM;
	unsigned int  	*mpNoVecsDrawn;


	const char* mOutputDir;

	std::vector<int>* mpIndexListOfVecsForSelection;

	friend class McSvmApproximationErrorFunction;
	friend class McSvmApproximationErrorFunctionGlobal;
};


//! \brief Approximation of Multi-class Support Vector Machines (SVMs)
class McSvmApproximationModel : public Model
{
public:
	McSvmApproximationModel(Array<double>& initVec){parameter = initVec;};
	virtual void model(const Array<double>& input, Array<double>& output){};
};


//! \brief Approximation of Multi-class Support Vector Machines (SVMs)
class McSvmApproximationErrorFunction : public ErrorFunction
{
public:
	McSvmApproximationErrorFunction(McSvmApproximation* svmApprox);

	double error(Model& model, const Array<double>& input, const Array<double>& target);
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	McSvmApproximation* mpApproxSVM;
};


//! \brief Approximation of Multi-class Support Vector Machines (SVMs)
class McSvmApproximationErrorFunctionGlobal : public ErrorFunction
{
public:
	McSvmApproximationErrorFunctionGlobal(McSvmApproximation* svmApprox);

	double error(Model& model, const Array<double>& input, const Array<double>& target);
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	McSvmApproximation *mpApproxSVM;
};


#endif

