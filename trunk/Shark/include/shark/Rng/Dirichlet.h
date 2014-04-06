/*!
 * 
 *
 * \brief       Implements a dirichlet distribution.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-01-01
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
#ifndef SHARK_RNG_DIRICHLET_H
#define SHARK_RNG_DIRICHLET_H

#include <shark/Rng/Gamma.h>
#include <shark/Rng/Rng.h>


#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>

#include <cmath>
#include <vector>

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
#include <iostream>
#endif

namespace shark{

	//! \brief Dirichlet distribution
	template<class RealType=double>
	class Dirichlet_distribution
	{
	public:
		typedef RealType input_type;
		typedef std::vector<RealType> result_type;

		explicit Dirichlet_distribution(size_t n=3,RealType alpha=1)
			:alphas_(n,alpha)
		{}
		explicit Dirichlet_distribution(const std::vector<RealType>& alphas)
			:alphas_(alphas)
		{}

		const std::vector<RealType>& alphas() const
		{
			return alphas_;
		}

		void reset() { }

		template<class Engine>
		result_type operator()(Engine& eng)const
		{
			unsigned n = alphas_.size();
			RealType sum = 0;
			std::vector<double> x;
			x.resize(n);
			for(size_t i=0; i<n; i++)
			{
				Gamma_distribution<> gamma(alphas_[i], 1.);
				x[i] = gamma(eng);
				sum += x[i];
			}
			for(size_t i=0; i<n; i++)
				x[i]/= sum;
			return x;
		}

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
		template<class CharT, class Traits>
		friend std::basic_ostream<CharT,Traits>&
			operator<<(std::basic_ostream<CharT,Traits>& os, const Dirichlet_distribution& d)
		{
			os << d.alphas.size();
			for(int i=0;i!=d.alphas_.size();++i)
				os << d.alphas_[i];
			return os;
		}

		template<class CharT, class Traits>
		friend std::basic_istream<CharT,Traits>&
			operator>>(std::basic_istream<CharT,Traits>& is, Dirichlet_distribution& d)
		{
			size_t size;
			is >> size;
			for(int i=0;i!=size;++i)
			{
				RealType element;
				is >> element;
				d.alphas_.push_back(element);
			}
			return is;
		}
#endif
	private:
		std::vector<RealType> alphas_;
	};

	/**
	* \brief Implements a Dirichlet distribution.
	* \tparam RngType The underlying generator type.
	*/
	template<typename RngType = shark::DefaultRngType>
	class Dirichlet:public boost::variate_generator<RngType*,Dirichlet_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,Dirichlet_distribution<> > Base;
	public:

		/**
		* \brief C'tor, associates the distribution with the given generator.
		* \param [in,out] rng Random number generator. 
		* \param [in] n Cardinality.
		* \param [in] alpha Support value.
		*/
		explicit Dirichlet(RngType& rng,size_t n=3,double alpha=1)
			:Base(&rng,Dirichlet_distribution<>(n,alpha))
		{}

		/**
		* \brief C'tor, associates the distribution with the given generator.
		* \param [in,out] rng Random number generator. 
		* \param [in] alphas Support values.
		*/
		explicit Dirichlet(RngType& rng,const std::vector<double>& alphas)
			:Base(&rng,Dirichlet_distribution<>(alphas))
		{}

		/** \brief Injects the default sampling operator. */
		using Base::operator();

		/**
		* \brief Creates a temporary instance of the distribution and samples it.
		* \param [in] n Cardinality.
		* \param [in] alpha Support value.
		*/
		std::vector<double> operator()(size_t n,double alpha) {
			Dirichlet_distribution<> dist(n,alpha);
			return dist(Base::engine());
		}

		/**
		* \brief Creates a temporary instance of the distribution and samples it.
		* \param [in] alphas Support values.
		*/
		std::vector<double> operator()(const std::vector<double> & alphas) {
			Dirichlet_distribution<> dist(alphas);
			return dist(Base::engine());
		}

		/**
		* \brief Accesses the support values.
		*/
		const std::vector<double> alphas()const {
			return Base::distribution().alphas();
		}

		/**
		* \brief Adjusts the support values.
		* \param [in] newAlphas New support values.
		*/
		void alphas(const std::vector<double>& newAlphas) {
			Base::distribution()=Dirichlet_distribution<>(newAlphas);
		}

		/**
		* \brief Adjusts the support values.
		* \param [in] n New cardinality.
		* \param [in] alphas Support value.
		*/
		void alphas(size_t n,double alphas) {
			Base::distribution()=Dirichlet_distribution<>(n,alphas);
		}

		/**
		* \brief Calculates the probability of the observation x.
		*/
		double p(const std::vector<double> &x)const
		{
			double p = 1.;
			double sum = 0.;
			for(int i=0; i<alphas().size(); i++)
			{
				p *= pow(x[i], alphas()[i]-1) / boost::math::tgamma(alphas()[i]);
				sum += alphas()[i];
			}
			return p * boost::math::tgamma(sum);
		}

	};
}
#endif
