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


///\brief Returns the p-value of a single sample test
///
/// Computes the test statistic of the supplied test using the data
/// and computes the probability that the null hypothesis
/// would generate the same test statistic
///
/// \param test the test to perform
/// \param data the data sample to test
/// \param tail the tail, or side to evaluate
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

///\brief Returns the p-value of a two-sample test
///
/// Computes the test statistic of the supplied test using the data
/// and computes the probability that the null hypothesis
/// would generate the same test statistic
///
/// Be careful to choose a test fitting to the data, e.g. dependent data
/// needs a paired test.
///
/// \param test the test to perform
/// \param dataX the first sample to test
/// \param dataY the second sample to test
/// \param tail the tail, or side to evaluate
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


///\brief Returns whether the test is significant on the data at alpha-level
///
/// Equivalent to p(test,data,tail) < alpha
///
/// \param test the test to perform
/// \param data the data sample to test
/// \param tail the tail, or side to evaluate
/// \param alpha the Significance threshold
template<class Test>
bool isSignificant(Test const& test, RealVector const& data, Tail tail, double alpha){
	return p(test,data,tail) < alpha;
}

///\brief Returns whether the test is significant on the data at alpha-level
///
/// Equivalent to p(test,data,tail) < alpha
///
/// \param test the test to perform
/// \param dataX the first sample to test
/// \param dataY the second sample to test
/// \param tail the tail, or side to evaluate
/// \param alpha the Significance threshold
template<class Test>
bool isSignificant(Test const& test, RealVector const& dataX, RealVector const& dataY, Tail tail, double alpha){
	return p(test, dataX, dataY, tail) < alpha;
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
		return std::sqrt(double(n)) * (sampleMean - m_mean)/std::sqrt(var);
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

/// \brief Implements the Wilcoxon ranksum test
///
/// Given two ordinal random variables X and Y with arbitrary distributions,
/// the test checks whether P(X>Y) = P(Y > X). This is done by computing
/// the ranks of the samples of X and Y and computing the ranksum of X. 
/// For large sample sizes > 20, the resulting statistic is approximately normal distributed
/// and we can perform a test on the resulting z-statistic.
///
/// Warning: the current implementation is not suited for small sample sizes.
/// The normal approximation only holds with a decent number of values and
/// there is currently no continuity correction implemented.
class WilcoxonRankSumTest{
public:
	boost::math::normal testDistribution(RealVector const& dataX, RealVector const& dataY)const{
		return boost::math::normal(0.0,1.0);
	}
	
	double statistics(RealVector const& dataX, RealVector const& dataY)const{
		std::size_t nX = dataX.size();
		std::size_t nY = dataY.size();
		std::size_t n = nX + nY;
		RealVector dataXsorted = dataX;
		RealVector dataYsorted = dataY;
		std::sort(dataXsorted.begin(),dataXsorted.end());
		std::sort(dataYsorted.begin(),dataYsorted.end());
		
		double ranksumX = -(nX*(nX+1.0)/2);
		double var = n +1;
		for(double x: dataX){
			//we compute the ranksum, taking ties into account
			//lower bound gives us the first element that is >=x.
			//upper bound gives us the first element that is >x
			//thus upper bound - lower bound gives the number of ties
			//the rank for ties is the average rank of a random ordering of all ties
			//therefore it is (upper bound + lower bound)/2
			//we have to compute this for both ranges
			auto rankXlower = std::lower_bound(dataXsorted.begin(),dataXsorted.end(),x) -dataXsorted.begin();
			auto rankXupper = std::upper_bound(dataXsorted.begin(),dataXsorted.end(),x) - dataXsorted.begin();
			auto rankYlower = std::lower_bound(dataYsorted.begin(),dataYsorted.end(),x) - dataYsorted.begin();
			auto rankYupper = std::upper_bound(dataYsorted.begin(),dataYsorted.end(),x) - dataYsorted.begin();
			double numTiesX = rankXupper - rankXlower;//needed for variance correction
			double numTies = numTiesX + (rankYupper - rankYlower);
			double rank = 1 + rankXlower + rankYlower+ (numTies-1.0)/2;//if numTies=1 <-> no other point than x has that rank
			ranksumX += rank;
			
			//variance correction for ties. 
			//formula is (numTies^3- numTies)/(n*(n-1))
			//however we add this for all x with the same rank so we have to divide
			//by numTiesX
			var -= (numTies*numTies * numTies - numTies)/numTiesX/(n*(n-1.0));//0 if numTies==0
		}
		var *= nX * nY / 12.0;
		//compute the u statistic
		double y = ranksumX - nX * nY/2.0;//subtract the mean rank sum
		return y/std::sqrt(var); 
		
	}
};

/// \brief Implements the Wilcoxon signed rank test
///
/// Given two ordinal random variables X and Y where X and Y are statistically dependent,
/// we can compute the variable Z=X- from the pairs.
/// the test checks whether P(Z > 0) = P(Z < 0). This is done by computing
/// the ranks R of |Z| and computing the signed ranksum E{sign(Z) R}. 
/// For large sample sizes > 20, the resulting statistic is approximately normal distributed
/// and we can perform a test on the resulting z-statistic.
///
/// Warning: the current implementation is not suited for small sample sizes.
/// The normal approximation only holds with a decent number of values and
/// there is currently no continuity correction implemented.
class WilcoxonSignedRankTest{
public:
	boost::math::normal testDistribution(RealVector const& dataX, RealVector const& dataY)const{
		return boost::math::normal(0.0,1.0);
	}
	double statistics(RealVector const& dataX, RealVector const& dataY)const{
		std::vector<double> diffs;
		for(std::size_t i = 0; i != dataX.size(); ++i){
			double diff = dataX[i] - dataY[i];
			if(diff == 0) continue;
			diffs.push_back(diff);
		}
		auto comp = [](double x, double y){return std::abs(x) < std::abs(y);};
		std::sort(diffs.begin(),diffs.end(), comp);
		
		std::size_t n = diffs.size();
		
		double W = 0.0;
		double var = n*(n+1.0)*(2*n+1.0)/6.0;
		for(double x: diffs){
			//we compute the ranksum, taking ties into account
			//lower bound gives us the first element that is >=x.
			//upper bound gives us the first element that is >x
			//thus upper bound - lower bound gives the number of ties
			//the rank for ties is the average rank of a random ordering of all ties
			//therefore it is (upper bound + lower bound)/2
			//we have to compute this for both ranges
			auto ranklower = std::lower_bound(diffs.begin(),diffs.end(),x, comp) - diffs.begin();
			auto rankupper = std::upper_bound(diffs.begin(),diffs.end(),x, comp) - diffs.begin();
			double numTies = rankupper - ranklower;
			double rank = 1 + ranklower + (numTies - 1.0)/2;
			double sign = x > 0? 1: -1;
			W += sign * rank;
			var -= (numTies*numTies * numTies - numTies)/numTies/12.0;//0 if numTies==0
		}
		return W/std::sqrt(var);
		
	}
};


 }}
 
 #endif