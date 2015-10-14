//===========================================================================
/*!
 * 
 *
 * \brief       unit test for Random Forest classifier
 * 
 * 
 * 
 * 
 *
 * \author      K. Hansen
 * \date        2012
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

#define BOOST_TEST_MODULE Models_RFClassifier
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_RFClassifier)

BOOST_AUTO_TEST_CASE( RF_Classifier ) {

	//Test data
	std::vector<RealVector> input(10, RealVector(2));
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
	input[6](0)=-4;
	input[6](1)=-3;
	input[7](0)=-2;
	input[7](1)=-1;
	input[8](0)=-6;
	input[8](1)=-8;
	input[9](0)=-2;
	input[9](1)=-2;

	std::vector<unsigned int> target(10);
	target[0]=0;
	target[1]=0;
	target[2]=1;
	target[3]=1;
	target[4]=2;
	target[5]=2;
	target[6]=3;
	target[7]=3;
	target[8]=4;
	target[9]=4;

	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	RFTrainer trainer;
	RFClassifier model;

	trainer.train(model, dataset);

	Data<RealVector> prediction = model(dataset.inputs());

	ZeroOneLoss<unsigned int, RealVector> loss;
	double error = loss.eval(dataset.labels(), prediction);

	std::cout << model.countAttributes() << std::endl;

	BOOST_CHECK(error == 0.0);

}

BOOST_AUTO_TEST_SUITE_END()
