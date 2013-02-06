//===========================================================================
/*!
 *
 *  \brief unit test for soft nearest neighbor classifier
 *
 *
 *  \author  T. Glasmachers
 *  \date    2011
 *
 *  \par Copyright (c) 2011:
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


#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE MODELS_SOFT_NEAREST_NEIGHBOR_CLASSIFIER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/Models/SoftNearestNeighborClassifier.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( SOFT_NEAREST_NEIGHBOR_CLASSIFIER ) {
	std::vector<RealVector> input(6, RealVector(2));
	input[0](0)=1;
	input[0](1)=3;
	input[1](0)=-1;
	input[1](1)=3;
	input[2](0)=1;
	input[2](1)=0;
	input[3](0)=-1;
	input[3](1)=0;
	input[4](0)=1;
	input[4](1)=-3;
	input[5](0)=-1;
	input[5](1)=-3;
	std::vector<unsigned int> target(6);
	target[0]=0;
	target[1]=0;
	target[2]=1;
	target[3]=1;
	target[4]=2;
	target[5]=2;

	ClassificationDataset dataset(input, target);

	DenseRbfKernel kernel(0.5);
	SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
	SoftNearestNeighborClassifier<RealVector> model(&algorithm, 3);

	Data<RealVector> prediction=model(dataset.inputs());
	for (size_t i = 0; i<6; ++i)
	{
		BOOST_CHECK_SMALL(prediction.element(i)(target[i]) - 2.0/3.0, 1e-12);
	}

	ZeroOneLoss<unsigned int, RealVector> loss;
	double error = loss.eval(dataset.labels(), prediction);
	BOOST_CHECK(error == 0.0);
}
