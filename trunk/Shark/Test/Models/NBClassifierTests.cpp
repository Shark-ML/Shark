//===========================================================================
/*!
 * 
 *
 * \brief       Test cases for Naive Bayes classifier
 * 
 * 
 * 
 *
 * \author      B. Li
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#define BOOST_TEST_MODULE NaiveBayesClassifierTestModule

#include "shark/LinAlg/Base.h"
#include "shark/Models/NBClassifier.h"
#include "shark/Rng/AbstractDistribution.h"
#include "shark/Rng/Normal.h"

#include <boost/assign/list_of.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>

namespace shark {

/// Fixture for testing naive Bayes classifier
class NBClassifierFixture
{
public:

	typedef NBClassifier<>::AbstractDistPtr AbstractDistPtr;

	/// Create a normal distribution from given @a mean and @a variance
	AbstractDistPtr createNormalDist(double mean, double variance) const
	{
		return AbstractDistPtr(new Normal<>(Rng::globalRng, mean, variance));
	}
};

BOOST_FIXTURE_TEST_SUITE(NaiveBayesClassifierTests, NBClassifierFixture)

BOOST_AUTO_TEST_CASE(Test)
{
	NBClassifier<>::FeatureDistributionsType featureDists;
	// class 0
	std::vector<AbstractDistPtr> class0;
	class0.push_back(createNormalDist(5.855, 3.5033e-02));	// feature 0
	class0.push_back(createNormalDist(176.25, 1.2292e+02));	// feature 1
	class0.push_back(createNormalDist(11.25, 9.1667e-01));	// feature 2
	featureDists.push_back(class0);

	// class 1
	std::vector<AbstractDistPtr> class1;
	class1.push_back(createNormalDist(5.4175, 9.7225e-02)); // feature 0
	class1.push_back(createNormalDist(132.5, 5.5833e+02));  // feature 1
	class1.push_back(createNormalDist(7.5, 1.6667e+00));	// feature 2
	featureDists.push_back(class1);

	NBClassifier<> m_NBClassifier(featureDists);

	// add prior class distribution
	m_NBClassifier.setClassPrior(0, 0.5);
	m_NBClassifier.setClassPrior(1, 0.5);

	// Test that feature/class numbers can be get from the classifier
	std::size_t classSize;
	std::size_t featureSize;
	boost::tie(classSize, featureSize) = m_NBClassifier.getDistSize();
	BOOST_CHECK_EQUAL(classSize, 2u);
	BOOST_CHECK_EQUAL(featureSize, 3u);

	// Test that an individual distribution can be fetched from the classifier
	const double tolerancePercentage = 0.01;
	AbstractDistribution& dist = m_NBClassifier.getFeatureDist(0u, 0u);
	Normal<> normal = dynamic_cast<Normal<>&>(dist); // this should not throw exception
	BOOST_CHECK_CLOSE(normal.mean(), 5.855, tolerancePercentage);
	BOOST_CHECK_CLOSE(normal.variance(), 3.5033e-02, tolerancePercentage);

	// Finally, let us try to predict a sample
	RealVector sample(3);
	sample[0] = 6.0;
	sample[1] = 130.0;
	sample[2] = 8.0;
	BOOST_CHECK_EQUAL(m_NBClassifier(sample), 1u);
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace shark {
