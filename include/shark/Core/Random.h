/*!
 * 
 *
 * \brief       Shark Random number generation
 * 
 * 
 *
 * \author      O.Krause
 * \date        2017
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
#ifndef SHARK_CORE_RANDOM_H
#define SHARK_CORE_RANDOM_H

#include <random>
#include <shark/Core/DLLSupport.h>

namespace shark{
namespace random{

	/** \brief Default RNG of the shark library. */
	typedef std::mt19937 rng_type;
	
	SHARK_EXPORT_SYMBOL extern rng_type globalRng;
	
	///\brief Flips a coin with probability of heads being pHeads by drawing random numbers from rng.
	template<class RngType>
	bool coinToss(RngType& rng, double pHeads = 0.5){
		std::bernoulli_distribution dist(pHeads);
		return dist(rng);
	}
	
	///\brief Draws a number uniformly in [lower,upper] by drawing random numbers from rng.
	template<class RngType>
	double uni(RngType& rng, double lower = 0.0, double upper = 1.0){
		std::uniform_real_distribution<double> dist(lower,upper);
		return dist(rng);
	}
	
	///\brief Draws a discrete number in {low,low+1,...,high} by drawing random numbers from rng.
	template<class RngType, class T>
	T discrete(RngType& rng, T low, T high){
		std::uniform_int_distribution<T> dist(low, high);
		return dist(rng);
	}
	
	///\brief Draws a number from the normal distribution with given mean and variance by drawing random numbers from rng.
	template<class RngType>
	double gauss(RngType& rng, double mean = 0.0, double variance = 1.0){
		std::normal_distribution<double> dist(mean,std::sqrt(variance));
		return dist(rng);
	}
	
	///\brief Draws a number from the log-normal distribution as exp(gauss(m,v)) 
	template<class RngType>
	double logNormal(RngType& rng, double m = 0.0, double v = 1.0){
		return std::exp(gauss(rng,m,v));
	}

	///\brief draws a number from the truncated exponential distribution
	///
	/// draws from the exponential distribution p(x|lambda)= 1/Z exp(-lambda*x) subject to x < maximum
	/// as optuonal third argument it is possible to return the precomputed value of Z.
	template<class RngType>
	double truncExp(RngType& rng, double lambda, double maximum, double Z = -1.0){
		double y = uni(rng,0.0,maximum);
		if(lambda == 0){
			return y;
		}
		if(Z < 0)
			Z = 1-std::exp(-lambda*maximum);
		return - std::log(1. - y*Z)/lambda;
	}
	
	


}}

#endif 
