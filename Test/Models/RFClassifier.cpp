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

#include <sstream>


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
	ZeroOneLoss<unsigned int, RealVector> loss2;
	double error_train = loss.eval(train.labels(), model(train.inputs()));
	double error_train2 = loss2.eval(train.labels(), model.decisionFunction()(train.inputs()));
	double error_test = loss.eval(test.labels(), model(test.inputs()));
	
	BOOST_CHECK(error_train < 0.01);
	BOOST_CHECK_CLOSE(error_train2, error_train,0.001);
	BOOST_REQUIRE_EQUAL(model.numberOfModels(), 100);
	BOOST_REQUIRE_EQUAL(model.featureImportances().size(), 10);
	BOOST_CHECK_SMALL(std::abs(error_test - model.OOBerror()), 0.02);
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK(model.featureImportances()(i) > 0.01);
		BOOST_CHECK(model.featureImportances()(i+5) < 0.01);
	}
	
	
	//serialisation test
	std::string str;
	
	{
		std::ostringstream outputStream;
		TextOutArchive oa(outputStream);  
		oa << model;
		str = outputStream.str();
	}
	//and create a new model from the serialization
	RFClassifier<unsigned int> modelDeserialized;
	std::istringstream inputStream(str);  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	double error_train_serialized = loss.eval(train.labels(), modelDeserialized(train.inputs()));
	double error_train2_serialized = loss2.eval(train.labels(), modelDeserialized.decisionFunction()(train.inputs()));
	double error_test_serialized = loss.eval(test.labels(), modelDeserialized(test.inputs()));
	BOOST_REQUIRE_CLOSE(error_train_serialized, error_train, 1.e-13);
	BOOST_REQUIRE_CLOSE(error_train2_serialized, error_train2, 1.e-13);
	BOOST_REQUIRE_CLOSE(error_test_serialized, error_test, 1.e-13);

	// Find the leaf for a sample
	std::vector<unsigned int> nodeIds;
	for(std::size_t m=0; m<model.numberOfModels(); ++m){
		auto aTree = model.model(m);
		auto result = aTree.findLeaf(test.inputs().element(0));
		nodeIds.push_back(result);
		BOOST_CHECK(result < aTree.numberOfNodes());
	}
}

BOOST_AUTO_TEST_SUITE_END()
