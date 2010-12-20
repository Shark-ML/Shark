//===========================================================================
/*!
 *  \file McSvm.cpp
 *
 *  \brief Multi class SVM example
 *
 *  \author  T. Glasmachers
 *  \date    2008, 2010
 *
 *  \par Copyright (c) 1999-2010:
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
class TestProblem : public DataSource
{
public:
	TestProblem(unsigned int classes = 5, double variance = 1.0)
	{
		this->classes = classes;
		this->variance = variance;

		dataDim = 2;				// input space dimension
		targetDim = 1;				// output is just a single number in {0, ..., classes-1}
	}

	~TestProblem() { }


	bool GetData(Array<double>& data, Array<double>& target, int count)
	{
		data.resize(count, dataDim, false);
		target.resize(count, 1, false);

		int i, c;
		for (i=0; i<count; i++)
		{
			c = Rng::discrete(0, classes - 1);
			data(i, 0) = 2.0 * cos((c * M_2PI) / classes) + Rng::gauss(0.0, variance);
			data(i, 1) = 2.0 * sin((c * M_2PI) / classes) + Rng::gauss(0.0, variance);
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
	double variance = 1.0;			// noise variance
	TestProblem source(classes, variance);

	// sample training and test dataset
	Dataset dataset;
	dataset.CreateFromSource(source, 500, 10000);

	// setup the kernel and the classifiers
	double C = 0.01;
	RBFKernel kernel(0.5);
	MultiClassSVM aio_svm(&kernel, classes);
	MultiClassSVM cs_svm(&kernel, classes);
	MultiClassSVM llw_svm(&kernel, classes);
	MultiClassSVM dgi_svm(&kernel, classes);
	MultiClassSVM ova_svm(&kernel, classes);
	MultiClassSVM occ_svm(&kernel, classes);
	MultiClassSVM ebcs_svm(&kernel, classes);

	// train the machines
	printf("\nMACHINE TRAINING\n");
	{
		printf("training all-in-one machine ..."); fflush(stdout);
		AllInOneMcSVM meta(&aio_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(aio_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training crammer & singer machine ..."); fflush(stdout);
		CrammerSingerMcSVM meta(&cs_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(cs_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training lee, lin & wahba machine ..."); fflush(stdout);
		LLWMcSVM meta(&cs_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(llw_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training dogan, glasmachers & igel machine ..."); fflush(stdout);
		DGIMcSVM meta(&cs_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(dgi_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training one-versus-all machine ..."); fflush(stdout);
		OVAMcSVM meta(&ova_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(ova_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training one-class-cost machine ..."); fflush(stdout);
		OCCMcSVM meta(&occ_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(occ_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}
	{
		printf("training epoch-based crammer-and-singer machine ..."); fflush(stdout);
		EpochBasedCsMcSvm meta(&ebcs_svm, C);
		SVM_Optimizer svmopt;
		svmopt.init(meta);
		svmopt.optimize(ebcs_svm, dataset.getTrainingData(), dataset.getTrainingTarget());
		printf(" done.\n");
	}

	// evaluate all machines on the test dataset
	printf("\nPREDICTION\n");
	ZeroOneLoss loss;
	double aio_err = loss.error(aio_svm, dataset.getTestData(), dataset.getTestTarget());
	double cs_err = loss.error(cs_svm, dataset.getTestData(), dataset.getTestTarget());
	double llw_err = loss.error(llw_svm, dataset.getTestData(), dataset.getTestTarget());
	double dgi_err = loss.error(dgi_svm, dataset.getTestData(), dataset.getTestTarget());
	double ova_err = loss.error(ova_svm, dataset.getTestData(), dataset.getTestTarget());
	double occ_err = loss.error(occ_svm, dataset.getTestData(), dataset.getTestTarget());
	double ebcs_err = loss.error(ebcs_svm, dataset.getTestData(), dataset.getTestTarget());

	printf("\nEVALUATION\n");
	printf("all-in-one machine:                 %g%% error\n", 100.0 * aio_err);
	printf("crammer & singer machine:           %g%% error\n", 100.0 * cs_err);
	printf("lee, lin & wahba machine:           %g%% error\n", 100.0 * llw_err);
	printf("dogan, glasmachers & igel machine:  %g%% error\n", 100.0 * dgi_err);
	printf("one-versus-all machine:             %g%% error\n", 100.0 * ova_err);
	printf("one-class-cost machine:             %g%% error\n", 100.0 * occ_err);
	printf("epoch-based c & s machine:          %g%% error\n", 100.0 * ebcs_err);
	printf("\n");
}
