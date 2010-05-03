//===========================================================================
/*!
 *  \file McSvm.cpp
 *
 *  \brief Multi class SVM example
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 1999-2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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


#include <ReClaM/Dataset.h>
#include <ReClaM/ClassificationError.h>
#include <ReClaM/Svm.h>


// definition of a simple test problem
class MultiClassTestProblem : public DataSource
{
public:
	MultiClassTestProblem(unsigned int classes = 5, double variance = 1.0)
	{
		this->classes = classes;
		this->variance = variance;

		dataDim = 2;				// input space dimension
		targetDim = 1;				// output is just a single number in {0, ..., classes-1}
	}

	~MultiClassTestProblem() { }


	bool GetData(Array<double>& data, Array<double>& target, int count)
	{
		data.resize(count, dataDim, false);
		target.resize(count, 1, false);

		int i, c;
		for (i=0; i<count; i++)
		{
			c = Rng::discrete(0, classes - 1);
			data(i, 0) = 2.0 * cos(c * M_2PI / classes) + Rng::gauss(0.0, variance);
			data(i, 1) = 2.0 * sin(c * M_2PI / classes) + Rng::gauss(0.0, variance);
			target(i, 0) = (double)c;
		}

		return true;
	}

protected:
	unsigned int classes;
	double variance;
};


int main(int argc, char* argv[])
{
	printf("\nMulti class support vector machine example program\n");

	// define the test problem
	unsigned int classes = 5;		// number of classes
	double variance = 4.0;			// noise variance
	MultiClassTestProblem source(classes, variance);

	// sample training and test dataset
	Dataset dataset(source, 500, 10000);

	// setup the kernel and the classifiers
	RBFKernel kernel(0.5);
	MultiClassSVM aio_svm(&kernel, classes, true);
	MultiClassSVM cs_svm(&kernel, classes, true);
	MultiClassSVM ova_svm(&kernel, classes, true);
	MultiClassSVM occ_svm(&kernel, classes, true);

	// train the machines
	printf("\nMACHINE TRAINING\n");
	{
		printf("training all-in-one machine ..."); fflush(stdout);
		double C = 0.01;
		AllInOneMcSVM meta(&aio_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(aio_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training crammer & singer machine ..."); fflush(stdout);
		double beta = 2.0 / 0.01;
		CrammerSingerMcSVM meta(&cs_svm, beta);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(cs_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training one-versus-all machine ..."); fflush(stdout);
		double C = 0.01;
		OVAMcSVM meta(&ova_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(ova_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training one-class-cost machine ..."); fflush(stdout);
		double C = 0.01;
		OCCMcSVM meta(&occ_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(occ_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}

	// evaluate all machines on the test dataset
	printf("\nPREDICTION\n");
	ZeroOneLoss loss;
	double aio_err = loss.error(aio_svm, dataset.getTestData(), dataset.getTestTarget());
	double cs_err = loss.error(cs_svm, dataset.getTestData(), dataset.getTestTarget());
	double ova_err = loss.error(ova_svm, dataset.getTestData(), dataset.getTestTarget());
	double occ_err = loss.error(occ_svm, dataset.getTestData(), dataset.getTestTarget());

	printf("\nEVALUATION\n");
	printf("all-in-one machine:        %g%% error\n", 100.0 * aio_err);
	printf("crammer & singer machine:  %g%% error\n", 100.0 * cs_err);
	printf("one-versus-all machine:    %g%% error\n", 100.0 * ova_err);
	printf("one-class-cost machine:    %g%% error\n", 100.0 * occ_err);
	printf("\n");
}
