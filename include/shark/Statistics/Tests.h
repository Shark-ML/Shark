/*!
 * 
 *
 * \brief       Calculate statistics given a range of values.
 * 
 * 
 *
 * \author     O.Krause
 * \date        2015
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
#ifndef SHARK_STATISTICS_TESTS_H
#define SHARK_STATISTICS_TESTS_H

#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/normal.hpp>
#include <shark/LinAlg/Base.h>
namespace shark{namespace statistics{

enum class Tail{
	Left,
	Right,
	TwoSided
};

template<class Test>
double p(Test const& test, RealVector const& data, Tail tail){
	auto dist = test.testDistribution(data);
	double t = test.statistics(data);
	if(tail == Tail::Left){
		return cdf(dist, t);
	}else if( tail == Tail::Right){
		return cdf(complement(dist, t));
	}else{
		return 2 * cdf(complement(dist,std::abs(t)));
	}
}

template<class Test>
double p(Test const& test, RealVector const& dataX, RealVector const& dataY, Tail tail){
	auto dist = test.testDistribution(dataX,dataY);
	double t = test.statistics(dataX,dataY);
	if(tail == Tail::Left){
		return cdf(dist, t);
	}else if( tail == Tail::Right){
		return cdf(complement(dist, t));
	}else{
		return 2 * cdf(complement(dist, std::abs(t)));
	}
}

template<class Test>
bool isSignificant(Test const& test, RealVector const& data, Tail tail, double alpha){
	auto dist = test.testDistribution(data);
	double t = test.statistics(data);
	if(tail == Tail::Left){
		return t < quantile(dist, alpha);
	}else if( tail == Tail::Right){
		return t > quantile(complement(dist, alpha));
	}else{
		return std::abs(t) > quantile(complement(dist, alpha/2));
	}
}

template<class Test>
bool isSignificant(Test const& test, RealVector const& dataX, RealVector const& dataY, Tail tail, double alpha){
	auto dist = test.testDistribution(dataX, dataY);
	double t = test.statistics(dataX, dataY);
	if(tail == Tail::Left){
		return t < quantile(dist, alpha);
	}else if( tail == Tail::Right){
		return t > quantile(complement(dist), alpha);
	}else{
		return std::abs(t) > quantile(complement(dist), alpha/2);
	}
}

/// \brief Class conducting a one-sample student's t-test
///
/// Assume the random variable X being normally distributed.
/// The one sample t-test answers the question: 
/// Is mean(X)=mu?
/// student's t-test does not make any assumption on the variance of X
/// therefore its test statistic follows a student's t-distribution.
class TTest{
public:
	/// Construct a TTest for a given target mean
	///
	/// \param mean The target mean to test against
	TTest(double mean):m_mean(mean){}

	boost::math::students_t testDistribution(RealVector const& data)const{
		return boost::math::students_t(data.size() - 1);
	}
	double statistics(RealVector const& data)const{
		//compute test statistic
		std::size_t n = data.size();
		double sampleMean = sum(data)/n;
		double var = sum(sqr(data - sampleMean))/(n-1.0);
		double std = std::sqrt(var);
		return std::sqrt(double(n)) * (sampleMean - m_mean)/std;
	}
private:
	double m_mean;
};

/// \brief Class conducting a two-sample t-test with dependent samples
///
/// Assume two random variables X and Y being normally distributed and
/// X and Y are statistically dependent.
/// The paired sample t-test answers the question: 
/// Is mean(X) = mean(Y)?
class PairedTTest{
public:
	boost::math::students_t testDistribution(RealVector const& dataX, RealVector const& dataY)const{
		return boost::math::students_t(dataX.size() - 1);
	}
	double statistics(RealVector const& dataX, RealVector const& dataY)const{
		TTest test(0);
		return test.statistics(dataX - dataY);
	}
};

/// \brief Class conducting a two-sample t-test
///
/// Assume two random variables X and Y being normally distributed.
/// The two sample t-test answers the question: 
/// Is mean(X) = mean(Y)?
/// The original paired student's t-test assumes that the variances of X and Y are the same.
/// however, when we drop this assumption we can still perform the test and the result
/// is known as the Welch's t-test. Welch's t-test is the default choice as it is more robust.
class TwoSampleTTest{
public:
	TwoSampleTTest(bool equalVariance=false):m_equalVariance(equalVariance){}
	boost::math::students_t testDistribution(RealVector const& dataX, RealVector const& dataY)const{
		std::size_t nX = dataX.size();
		std::size_t nY = dataY.size();
		if(m_equalVariance){
			return boost::math::students_t(nX  + nY - 2);
		}else{
			double meanX = sum(dataX)/nX;
			double meanY = sum(dataY)/nY;
			double varX = sum(sqr(dataX - meanX))/(nX-1.0);
			double varY = sum(sqr(dataY - meanY))/(nY-1.0);
			
			double nu= (varX/nX + varY/nY) * (varX/nX + varY/nY);
			nu /= varX * varX / (nX*nX*(nX-1)) + varY * varY / (nY*nY*(nY-1)); 
			return boost::math::students_t(nu);
		}
	}
	double statistics(RealVector const& dataX, RealVector const& dataY)const{
		//compute test statistic
		std::size_t nX = dataX.size();
		std::size_t nY = dataY.size();
		double meanX = sum(dataX)/nX;
		double meanY = sum(dataY)/nY;
		double varX = sum(sqr(dataX - meanX))/(nX-1.0);
		double varY = sum(sqr(dataY - meanY))/(nY-1.0);
		
		double var = varX/nX + varY/nY;
		if(m_equalVariance){
			var = ((nX - 1.0)* varX + (nY - 1.0) * varY) / (nX + nY - 2);
			var *= 1.0/nX + 1.0/nY;
		}
		
		return  (meanX - meanY) / std::sqrt(var); 
		
	}
private:
	bool m_equalVariance;
};

 }}
 
 #endif