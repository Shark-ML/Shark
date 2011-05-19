//===========================================================================
/*!
 *  \file McSvmApproximationExample.cpp
 *
 *  \brief Multi-class SVM approximation example
 *
 *  \author  B. Weghenkel
 *  \date    2011
 *
 *  \par Copyright (c) 1999-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#include <string>
#include <stdio.h>

#include <Rng/GlobalRng.h>
#include <ReClaM/Svm.h>
#include <ReClaM/McSvmApproximation.h>
#include <ReClaM/ClassificationError.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Dataset.h>


using namespace std;
using std::cout;
using std::endl;


// definition of a simple test problem
class TestProblem : public DataSource
{
public:
	TestProblem()
	{
		dataDim = 2;				// input space dimension
		targetDim = 1;				// output is just a single number in {0, ..., classes-1}
	}

	~TestProblem() { }


	bool GetData(Array<double>& data, Array<double>& target, int count)
	{
		data.resize(count, dataDim, false);
		target.resize(count, 1, false);

		for (int i = 0; i < count; i++) {

		  double x = Rng::uni(-1.0, 1.0);
      double y = Rng::uni(-1.0, 1.0);
      data(i, 0) = x;
      data(i, 1) = y;

      if(x<0)
      {
        if(y<0)
          target(i, 0) = 0;
        else
          target(i, 0) = 2;
      }
      else
      {
        if(y<0)
          target(i, 0) = 1;
        else
          target(i, 0) = 3;
      }
		}

		return true;
	}
};


int main(int argc, char** argv)
{
	// define the test problem
	unsigned int classes = 4;		// number of classes
	TestProblem source;

	// sample training and test dataset
	Dataset dataset;
	dataset.CreateFromSource(source, 5000, 10000);

	const Array<double>& trainData       = dataset.getTrainingData();
	const Array<double>& trainDataLabels = dataset.getTrainingTarget();
	const Array<double>& testData        = dataset.getTestData();
	const Array<double>& testDataLabels  = dataset.getTestTarget();

	//
	double C     = 100.0;
	double gamma = 1.0;
	RBFKernel kernel(gamma);
	MultiClassSVM mc_svm(&kernel, classes);
	AllInOneMcSVM meta_svm(&mc_svm, C);
	SVM_Optimizer SVMopt;
	SVMopt.init(meta_svm);
	ZeroOneLoss loss;

	cout << "SVM training\n";
	SVMopt.optimize(mc_svm, trainData, trainDataLabels);

	// calculate #SVs
	unsigned noSV = 0;
	for( unsigned i = 0; i < mc_svm.getExamples(); i++ )
		for( unsigned c = 0; c < classes; c++ )
			if( mc_svm.getAlpha(i,c) != 0.0 ) {
			  //cout << "alpha: " << mc_svm.getAlpha(i,c) << endl;
				noSV++;
			}

	// calculate # of examples used as SV
	unsigned noExamplesUsedAsSV = 0;
	for( unsigned i = 0; i < mc_svm.getExamples(); i++ )
		for( unsigned c = 0; c < classes; c++ )
			if( mc_svm.getAlpha(i,c) != 0.0 ) {
				noExamplesUsedAsSV++;
				break;
			}

	cout << "# of Examples used as SVs: " << noExamplesUsedAsSV << " (" << noSV << ")" << endl;

	cout << "classify test data" << endl;
	cout << "accuracy: " << 100.0 *(1.0 - loss.error(mc_svm, testData, testDataLabels)) << "%\n";

	// approximate
  McSvmApproximation* approximateSVM = new McSvmApproximation(&mc_svm, true);
  approximateSVM->setTargetNoVecsForApproximatedSVM(50);
	approximateSVM->approximate(testData, testDataLabels);

	// evaluate
  MultiClassSVM* approxSvm = approximateSVM->getApproximatedSVM();
	double error = loss.error(*approxSvm, testData, testDataLabels);
	cout << "accuracy: " << 100.0 *(1.0 - error) << "%\n";

//  // final gradient descent on approximated model
//  approximateSVM->setNoGradientDescentIterations(100);
//	svmApproximationError = approximateSVM->gradientDescent();
//  cout << "accuracy: " << 100.0 *(1.0 - loss.error(*approxSvm, testData, testDataLabels)) <<"%\n";

	delete approximateSVM;
}
