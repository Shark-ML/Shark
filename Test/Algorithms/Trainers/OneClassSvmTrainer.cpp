//===========================================================================
/*!
 *  \brief test case for the one-class-SVM
 *
 *  \author T. Glasmachers
 *  \date 2013
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
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_ONE_CLASS_SVM
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/OneClassSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;


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


BOOST_AUTO_TEST_CASE( ONE_CLASS_SVM_TEST )
{
	const std::size_t ell = 2500;
	const double nu = 0.7;
	const double gamma = 0.5;
	const double threshold = 0.02;   // 1 / sqrt(ell)

	GaussianRbfKernel<> kernel(gamma);
	KernelExpansion<RealVector> ke;

	Gaussians problem;
	UnlabeledData<RealVector> data = problem.generateDataset(ell);

	OneClassSvmTrainer<RealVector> trainer(&kernel, nu);
	trainer.sparsify() = false;
	trainer.train(ke, data);

	Data<RealVector> output = ke(data);

	// check deviation of fraction of negatives from nu
	std::size_t pos = 0;
	std::size_t neg = 0;
	for (std::size_t i=0; i<ell; i++)
	{
		double f = output.element(i)(0);
		if (f > 0.0) pos++;
		else if (f < 0.0) neg++;
	}

	double p = (double)pos / (double)ell;
	double n = (double)neg / (double)ell;
	BOOST_CHECK_SMALL(p - 1.0 + nu, threshold);
	BOOST_CHECK_SMALL(n - nu, threshold);
}
