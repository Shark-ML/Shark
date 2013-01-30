//===========================================================================
/*!
 *
 *  \brief unit test for CART classifier
 *
 *
 *
 *  \author  K. Hansen
 *  \date    2012
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
#define BOOST_TEST_MODULE Models_CARTClassifier
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/CARTTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( CART_Classifier ) {

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

	ClassificationDataset dataset(input, target);

	CARTTrainer trainer;
	CARTClassifier<RealVector> model;

	trainer.train(model, dataset);

	Data<RealVector> prediction = model(dataset.inputs());
	ZeroOneLoss<unsigned int, RealVector> loss;
	double error = loss.eval(dataset.labels(), prediction);

	BOOST_CHECK(error == 0.0);

}
