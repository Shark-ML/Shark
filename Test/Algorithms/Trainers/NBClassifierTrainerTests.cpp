//===========================================================================
/*!
 * 
 *
 * \brief       Test cases for naive Bayes classifier trainer
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
#define BOOST_TEST_MODULE NaiveBayesClassifierTrainerTestModule

#include "shark/Algorithms/Trainers/Distribution/DistTrainerContainer.h"
#include "shark/Algorithms/Trainers/Distribution/NormalTrainer.h"
#include "shark/Algorithms/Trainers/NBClassifierTrainer.h"
#include "shark/Data/Csv.h"
#include "shark/LinAlg/Base.h"
#include "shark/Models/NBClassifier.h"
#include "shark/Rng/Normal.h"
#include "shark/Rng/Rng.h"
#include "shark/Rng/Uniform.h"

#include <boost/assign/list_of.hpp>
#include <boost/range/algorithm/equal.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>

namespace shark {

/// A help class to help test cases access internal states of NBClassifier
class DummyNBClassifier : public NBClassifier<>
{
public:

	typedef NBClassifier<> Base;

	explicit DummyNBClassifier(const NBClassifier<>::FeatureDistributionsType& featureDists)
	:
		NBClassifier<>(featureDists),
		m_featureDistributions(Base::m_featureDistributions),
		m_classPriors(Base::m_classPriors)
	{}

	DummyNBClassifier(std::size_t classSize, std::size_t featureSize)
	:
		NBClassifier<>(classSize, featureSize),
		m_featureDistributions(Base::m_featureDistributions),
		m_classPriors(Base::m_classPriors)
	{}

	typedef NBClassifier<>::FeatureDistributionsType FeatureDistributionsType;
	typedef NBClassifier<>::ClassPriorsType ClassPriorsType;
	FeatureDistributionsType& m_featureDistributions;
	ClassPriorsType& m_classPriors;
};

/// Fixture for testing naive Bayes classifier trainer
class NBClassifierTrainerFixture
{
public:

  typedef NBClassifier<>::AbstractDistPtr AbstractDistPtr;

  NBClassifierTrainerFixture() : tolerancePercentage(0.01) {
	csvStringToData(m_labeledData,m_dataInString,LAST_COLUMN);
  }

  void verifyNormalDistribution(const DummyNBClassifier& myNbClassifier)
  {
      // Verify class distribution
      BOOST_CHECK_EQUAL(myNbClassifier.m_classPriors.size(), 2u);
      BOOST_CHECK_CLOSE(myNbClassifier.m_classPriors[0], 0.5, tolerancePercentage);
      BOOST_CHECK_CLOSE(myNbClassifier.m_classPriors[1], 0.5, tolerancePercentage);

      // Verify feature distribution. format(mean, variance)
      BOOST_REQUIRE(myNbClassifier.m_featureDistributions.size() == 2u);

      // male
      {
          std::vector<std::pair<double, double> > expectedMaleDist =
              boost::assign::map_list_of(5.855, 3.5033e-02)(176.25,1.2292e+02)(11.25,9.1667e-01);
          unsigned int i = 0;
          BOOST_REQUIRE_EQUAL(myNbClassifier.m_featureDistributions[0].size(), 3u);
          BOOST_FOREACH(const NBClassifier<>::AbstractDistPtr& dist, myNbClassifier.m_featureDistributions[0])
          {
              const Normal<>& featureDistribution = dynamic_cast<const Normal<>&>(*dist);
              BOOST_CHECK_CLOSE(featureDistribution.mean(), expectedMaleDist[i].first, tolerancePercentage);
              BOOST_CHECK_CLOSE(featureDistribution.variance(), expectedMaleDist[i].second, tolerancePercentage);
              ++i;
          }
      }

      // female
      {
          std::vector<std::pair<double, double> > expectedFemaleDist =
              boost::assign::map_list_of(5.4175,9.7225e-02)(132.5,5.5833e+02)(7.5,1.6667e+00);
          unsigned int i = 0;
          BOOST_REQUIRE_EQUAL(myNbClassifier.m_featureDistributions[1].size(), 3u);
          BOOST_FOREACH(const NBClassifier<>::AbstractDistPtr& dist, myNbClassifier.m_featureDistributions[1])
          {
              const Normal<>& featureDistribution = dynamic_cast<const Normal<>&>(*dist);
              BOOST_CHECK_CLOSE(featureDistribution.mean(), expectedFemaleDist[i].first, tolerancePercentage);
              BOOST_CHECK_CLOSE(featureDistribution.variance(), expectedFemaleDist[i].second, tolerancePercentage);
              ++i;
          }
      }
  }

  /// Create a normal distribution from given @a mean and @a variance
  AbstractDistPtr createNormalDist() const
  {
      return AbstractDistPtr(new Normal<>(Rng::globalRng));
  }

  static const std::string m_dataInString;
  LabeledData<RealVector, unsigned int> m_labeledData;
  const double tolerancePercentage;

  /// The class under test
  NBClassifierTrainer<> m_NbTrainer;
};

// Last column is label, 0 is for male, 1 is for female
const std::string NBClassifierTrainerFixture::m_dataInString = "\
6,180,12,0\n\
5.92,190,11,0\n\
5.58,170,12,0\n\
5.92,165,10,0\n\
5,100,6,1\n\
5.5,150,8,1\n\
5.42,130,7,1\n\
5.75,150,9,1\r";

BOOST_FIXTURE_TEST_SUITE(NBClassiferTrainerTests, NBClassifierTrainerFixture)

BOOST_AUTO_TEST_CASE(TestNormal)
{
	// Test that the trainer is able to provide correct distributions to the classifier

	DummyNBClassifier myNbClassifier(2u, 3u); // all distributions are normal
	m_NbTrainer.train(myNbClassifier, m_labeledData);
	verifyNormalDistribution(myNbClassifier);
}

BOOST_AUTO_TEST_CASE(AccessInvidualDistTrainer)
{
	// Test that we are able to access individual distribution trainer

	// get
	DistTrainerContainer& trainerContainer = m_NbTrainer.getDistTrainerContainer();
	BOOST_CHECK_NO_THROW(trainerContainer.getNormalTrainer());

	// set
	BOOST_CHECK_NO_THROW(trainerContainer.setNormalTrainer(NormalTrainer()));
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace shark {
