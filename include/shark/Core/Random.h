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
#include <mutex>
#include <thread>
#include <unordered_set>

#include <iostream>
namespace shark{
namespace random{

	/// \brief RNG of the shark library.
	///
	/// The GlobalRng is a mostly threadsafe Rng based on mt19937.
	/// It can only be accessed using the free function globalRng()().
	/// There is one instance of the Rng per thread. Except seeding, 
	/// All operations on an individual instance
	/// of GlobalRng are threadsafe. Thus, during seeding it has to be ensured that no
	/// thread generates random numbers to prevent race-conditions.
	///
	/// The implementation uses a global registry of active instances of the rng.
	/// Each Rng registers itself and a call to seed reseeds eeach rng
	/// with numbers derived from the seed. However, the order in which this happens
	/// is unspecified. Threads created afterwards are also initialized based on this seed.
	/// If no seeding is performed, the initial seed starts
	/// from the standard mandated default seed for mt19937.
	class ThreadsafeRng : public std::mt19937{
	public:
		using result_type = std::mt19937::result_type;
	
		/// \brief Destructor
		~ThreadsafeRng(){
			std::lock_guard<std::mutex> lock(m_state->stateMutex);
			m_state->all.erase(this);
		}

		/// \brief Get Access to an instance of the rng
		///
		/// There is one instane per thread. Thus care has to be taken that one rng is not
		/// shared between threads.
		friend ThreadsafeRng& globalRng();
		

		
		/// \brief Reseeds globally all Rngs in all threads
		///
		/// This operation is not threadsafe!
		void seed(result_type s){
			std::lock_guard<std::mutex> lock(m_state->stateMutex);
			m_state->globalseed = s;
			for (ThreadsafeRng* p : m_state->all)
				p->seedInternal( m_state->globalseed++);//race condition with parallel thread using operator()
		}
	private:
		void seedInternal(result_type s){
			std::mt19937::seed(s);
		};
		struct GlobalState{
			std::mutex stateMutex;
			result_type globalseed;//next seed to use. guarded by stateMutex
			std::unordered_set<ThreadsafeRng*> all;//list of all rngs. guarded by stateMutex

			GlobalState(){
				globalseed = std::mt19937::default_seed;
			}
		};
		
		/// \brief private constructor. Can only be created by globalRng()();
		///
		/// Registers the Rng in the global list of Rngs and seeds it from the global state
		ThreadsafeRng(std::shared_ptr<GlobalState> const& state)
		: m_state(state){
			std::lock_guard<std::mutex> lock(m_state->stateMutex);
			seedInternal(m_state->globalseed++);
			m_state->all.insert(this);
			
		}
		std::shared_ptr<GlobalState> m_state;		
	};
	
	typedef ThreadsafeRng rng_type;
	
	/// \brief Get Access to an instance of the rng
	///
	/// There is one instane per thread. Thus care has to be taken that one rng is not
	/// shared between threads.
	inline rng_type& globalRng(){
		//We use a shared_ptr to prevent us from the static destructor order fiasco: 
		// if global state is killed before the thread_local, we try to delete dead objects. bad.
		// Through shared_ptr, nothing is cleaned up, until the last shared_ptr is dead.
		static auto state = std::make_shared<ThreadsafeRng::GlobalState>();
		static thread_local rng_type rng(state);
		return rng;
	}
	
	

	
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
