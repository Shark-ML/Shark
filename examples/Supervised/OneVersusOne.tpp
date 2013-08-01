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
#include <shark/Models/Converter.h>
#include <shark/Models/ConcatenatedModel.h>
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
	std::vector< KernelExpansion<RealVector>* > bin_ke(pairs);
	ThresholdConverter conv;
	std::vector< ConcatenatedModel<RealVector, unsigned int>* > bin_svm(pairs);
	for (std::size_t n=0, c=1; c<classes; c++)
	{
		std::vector< OneVersusOneClassifier<RealVector>::binary_classifier_type* > vs_c;
		for (std::size_t e=0; e<c; e++, n++)
		{
			//~ // create two-class sub-problem
			//~ std::vector<std::size_t> indices;
			//~ std::vector<unsigned int> binlabels;
			//~ for (std::size_t i=0; i<training.size(); i++)
			//~ {
				//~ unsigned int y = training.label(i);
				//~ if (y == e) { indices.push_back(i); binlabels.push_back(0); }
				//~ if (y == c) { indices.push_back(i); binlabels.push_back(1); }
			//~ }
			//~ UnlabeledData<RealVector> bininputs;
			//~ ((UnlabeledData<RealVector>)training).indexedSubset(indices, bininputs);
			//~ ClassificationDataset bindata(bininputs, binlabels);
			ClassificationDataset bindata = binarySubProblem(training,e,c);
				
			// train the binary machine
			CSvmTrainer<RealVector> trainer(&kernel, C);
			bin_ke[n] = new KernelExpansion<RealVector>(false);
			trainer.train(*bin_ke[n], bindata);
			bin_svm[n] = new ConcatenatedModel<RealVector, unsigned int>(bin_ke[n], &conv);
			vs_c.push_back(bin_svm[n]);
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

//	// clean up
//	for (std::size_t n=0; n<pairs; n++)
//	{
//		delete bin_ke[n];
//		delete bin_svm[n];
//	}
}
