//===========================================================================
/*!
 *  \brief Example program for the one-versus-one classifier based on SVMs
 *
 *  \author T. Glasmachers
 *  \date 2012
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================


#include <shark/Rng/GlobalRng.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Models/OneVersusOneClassifier.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <iostream>


using namespace shark;
using namespace std;


// data generating distribution for our toy
// multi-category classification problem
/// @cond EXAMPLE_SYMBOLS
class Problem : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	void draw(RealVector& input, unsigned int& label)const
	{
		label = Rng::discrete(0, 4);
		input.resize(1);
		input(0) = Rng::gauss() + 3.0 * label;
	}
};
/// @endcond

int main()
{
	// experiment settings
	unsigned int classes = 5;
	std::size_t ell = 100;
	std::size_t tests = 10000;
	double C = 10.0;
	double gamma = 0.5;

	// generate a very simple dataset with a little noise
	Problem problem;
	ClassificationDataset training = problem.generateDataset(ell);
	ClassificationDataset test = problem.generateDataset(tests);
	repartitionByClass(training);

	// kernel function
	GaussianRbfKernel<> kernel(gamma);
	// train the OVO machine
	OneVersusOneClassifier<RealVector> ovo;
	unsigned int pairs = classes * (classes - 1) / 2;
	std::vector< KernelClassifier<RealVector> > bin_svm(pairs);
	for (std::size_t n=0, c=1; c<classes; c++)
	{
		std::vector< OneVersusOneClassifier<RealVector>::binary_classifier_type* > vs_c;
		for (std::size_t e=0; e<c; e++, n++)
		{
			//get the binary subproblem
			ClassificationDataset bindata = binarySubProblem(training,e,c);
				
			// train the binary machine
			CSvmTrainer<RealVector> trainer(&kernel, C,false);
			trainer.train(bin_svm[n], bindata);
			vs_c.push_back(&bin_svm[n]);
		}
		ovo.addClass(vs_c);
	}

	// compute errors
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> output = ovo(training.inputs());
	double train_error = loss.eval(training.labels(), output);
	output = ovo(test.inputs());
	double test_error = loss.eval(test.labels(), output);
	cout << "training error: " << 100.0 * train_error << "%" << endl;
	cout << "    test error: " << 100.0 *  test_error << "%" << endl;
}
