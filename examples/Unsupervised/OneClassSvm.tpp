//===========================================================================
/*!
 * 
 *
 * \brief       One-Class Support Vector Machine example program.
 * 
 *  \par
 *  This program generates a toy data set composed of Gaussian
 *  distributions. It then uses a one-class SVM to model the
 *  densest regions. It visualizes the result.
 *
 * 
 *
 * \author      T. Glasmachers
 * \date        2013
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#include <shark/Algorithms/Trainers/OneClassSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/DataDistribution.h>

using namespace shark;
using namespace std;


class Gaussians : public DataDistribution<RealVector>
{
public:
	void draw(RealVector& point) const
	{
		point.resize(2);
		size_t cluster = Rng::discrete(0, 4);
		double alpha = 0.4 * M_PI * cluster;
		point(0) = 3.0 * cos(alpha) + 0.75 * Rng::gauss();
		point(1) = 3.0 * sin(alpha) + 0.75 * Rng::gauss();
	}
};


int main(int argc, char** argv)
{
	// experiment settings
	unsigned int ell = 100;     // number of training data point
	double nu = 0.5;            // probability mass to be covered, must fulfill 0 < mu < 1
	double gamma = 0.5;         // kernel bandwidth parameter

	GaussianRbfKernel<> kernel(gamma); // Gaussian kernel
	KernelExpansion<RealVector> ke; // (affine) linear function in kernel-induced feature space

	// generate artificial benchmark data
	Gaussians problem;
	UnlabeledData<RealVector> data = problem.generateDataset(ell);

	// define the learner
	OneClassSvmTrainer<RealVector> trainer(&kernel, nu);

	// train the model
	trainer.train(ke, data);

	// evaluate the model
	char output[35][71];
	RealVector input(2);
	for (std::size_t y=0; y<35; y++)
	{
		input(1) = 5.0 * (y - 17.0) / 17.0;
		for (std::size_t x=0; x<70; x++)
		{
			input(0) = 5.0 *  (x - 34.5) / 34.5;
			double val = ke(input)(0);
			output[y][x] = (val < 0.0) ? ' ' : ':';
		}
		output[y][70] = 0;
	}

	// mark the samples
	UnlabeledData<RealVector>::const_element_range elements = data.elements();
	for (UnlabeledData<RealVector>::const_element_range::const_iterator it = elements.begin(); it != elements.end(); ++it)
	{
		RealVector v = *it;
		int x = (int)std::floor(34.5 * v(0) / 5.0 + 34.5 + 0.5);
		int y = (int)std::floor(17.0 * v(1) / 5.0 + 17.0 + 0.5);
		if (x >= 0 && y >= 0 && x < 70 && y < 35) output[y][x] = '*';
	}

	// output to the console
	cout << endl
		<< "One-Class SVM example program." << endl
		<< "100 samples are drawn from a mixture of five Gaussians. Data samples" << endl
		<< "are marked with an asterisk '*'. The :::-shaded regions are the SVM's" << endl
		<< "estimate of the high-probability region of the distribution." << endl
		<< endl;
	for (std::size_t y=0; y<35; y++) cout << output[y] << endl;
	cout << endl;
}
