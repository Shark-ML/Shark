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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#include <shark/Data/DataDistribution.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_RFClassifier)

BOOST_AUTO_TEST_CASE( RF_Classifier ) {
	PamiToy generator(5,5,0,0.4);
	auto train = generator.generateDataset(200);
	auto test = generator.generateDataset(200);
	RFTrainer<unsigned int> trainer(true,true);
	RFClassifier<unsigned int> model;
	trainer.train(model, train);

	ZeroOneLoss<> loss;
	double error_train = loss.eval(train.labels(), model(train.inputs()));
	double error_test = loss.eval(test.labels(), model(test.inputs()));
	
	BOOST_REQUIRE_EQUAL(model.numberOfModels(), 100);
	BOOST_REQUIRE_EQUAL(model.featureImportances().size(), 10);
	BOOST_CHECK_SMALL(std::abs(error_test - model.OOBerror()), 0.02);
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK(model.featureImportances()(i) > 0.01);
		BOOST_CHECK(model.featureImportances()(i+5) < 0.01);
	}
}

BOOST_AUTO_TEST_SUITE_END()
