//===========================================================================
/*!
 * 
 *
 * \brief       Unit test for the generic one-versus-one classifier
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
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

#include <shark/Models/OneVersusOneClassifier.h>

#define BOOST_TEST_MODULE Models_OneVersusOneClassifier
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


// simple classifier for testing only
class ThresholdClassifier : public AbstractModel<double, unsigned int>
{
public:
	typedef AbstractModel<double, unsigned int> base_type;

	ThresholdClassifier(double threshold)
	: m_threshold(threshold)
	{ }

	std::string name() const
	{ return "ThresholdClassifier"; }

	RealVector parameterVector() const
	{
		RealVector p(1);
		p(0) = m_threshold;
		return p;
	}

	void setParameterVector(RealVector const& newParameters)
	{ m_threshold = newParameters(0); }

	std::size_t numberOfParameters() const
	{ return 1; }
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	void eval(BatchInputType const& x, BatchOutputType& y, State& state)const
	{ 
		y.resize(shark::size(x));
		for(std::size_t i = 0; i != shark::size(x); ++i){
			y(i) = (x(i) < m_threshold) ? 0 : 1;
		}
	}

protected:
	double m_threshold;
};


BOOST_AUTO_TEST_SUITE (Models_OneVersusOneClassifier)

BOOST_AUTO_TEST_CASE( Models_OneVersusOneClassifier )
{
	// Create a one-versus-one classifier for four classes.
	// It consists of 6 binary classifiers.
	ThresholdClassifier c10(0.5);
	ThresholdClassifier c20(1.0);
	ThresholdClassifier c21(1.5);
	ThresholdClassifier c30(1.5);
	ThresholdClassifier c31(2.0);
	ThresholdClassifier c32(2.5);
	std::vector<OneVersusOneClassifier<double>::binary_classifier_type*> l1(1);
	l1[0] = &c10;
	std::vector<OneVersusOneClassifier<double>::binary_classifier_type*> l2(2);
	l2[0] = &c20;
	l2[1] = &c21;
	std::vector<OneVersusOneClassifier<double>::binary_classifier_type*> l3(3);
	l3[0] = &c30;
	l3[1] = &c31;
	l3[2] = &c32;

	OneVersusOneClassifier<double> ovo;
	BOOST_CHECK_EQUAL(ovo.numberOfClasses(), 1);
	ovo.addClass(l1);
	BOOST_CHECK_EQUAL(ovo.numberOfClasses(), 2);
	ovo.addClass(l2);
	BOOST_CHECK_EQUAL(ovo.numberOfClasses(), 3);
	ovo.addClass(l3);
	BOOST_CHECK_EQUAL(ovo.numberOfClasses(), 4);

	// check parameters
	RealVector p = ovo.parameterVector();
	BOOST_CHECK_EQUAL(ovo.numberOfParameters(), 6);
	BOOST_CHECK_EQUAL(p.size(), 6);
	BOOST_CHECK_SMALL(std::fabs(p(0) - 0.5), 1e-14);
	BOOST_CHECK_SMALL(std::fabs(p(1) - 1.0), 1e-14);
	BOOST_CHECK_SMALL(std::fabs(p(2) - 1.5), 1e-14);
	BOOST_CHECK_SMALL(std::fabs(p(3) - 1.5), 1e-14);
	BOOST_CHECK_SMALL(std::fabs(p(4) - 2.0), 1e-14);
	BOOST_CHECK_SMALL(std::fabs(p(5) - 2.5), 1e-14);

	// create a simple data test set
	std::vector<double> inputs(4);
	inputs[0] = 0.0;
	inputs[1] = 1.0;
	inputs[2] = 2.0;
	inputs[3] = 3.0;
	std::vector<unsigned int> targets(4);
	targets[0] = 0;
	targets[1] = 1;
	targets[2] = 2;
	targets[3] = 3;
	LabeledData<double, unsigned int> dataset = createLabeledDataFromRange(inputs, targets);

	// check correctness of predictions
	for (std::size_t i=0; i<dataset.numberOfElements(); i++)
	{
		BOOST_CHECK_EQUAL(ovo(dataset.element(i).input), dataset.element(i).label);
	}
}

BOOST_AUTO_TEST_SUITE_END()
