/*!
 * 
 *
 * \brief       Very basic math abstraction layer.
 * 
 * \par
 * This file serves as a minimal abstraction layer.
 * Inclusion of this file makes some frequently used
 * functions, constants, and header file inclusions
 * OS-, compiler-, and version-independent.
 * 
 * 
 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_CORE_MATH_H
#define SHARK_CORE_MATH_H

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <limits>
#include <cmath>

namespace shark {

	// define doxygen group for shark global functions, var, etc. (should only appear once in all shark files)
	/**
	* \defgroup shark_globals shark_globals
	* 
	* \brief Several mathematical, linear-algebra, or other functions within Shark are not part of any particular class. They are collected here in the doxygen group "shark_globals"
	* 
	* @{
	*
	*/
	
	/**
	* \brief Constant for sqrt( 2 * pi ).
	*/
	static const double SQRT_2_PI = boost::math::constants::root_two_pi<double>();
//	static const double SQRT_PI = boost::math::constants::root_pi<double>();
	
	/// Maximum allowed input value for exp.
	template<class T>
	T maxExpInput(){
		return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::max_exponent;
	}
	/// Minimum value for exp(x) allowed so that it is not 0.
	template<class T>
	T minExpInput(){
		return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::min_exponent;
	}

	/// Calculates x^2.
	template <class T> 
	inline typename boost::enable_if<boost::is_arithmetic<T>, T>::type sqr( const T & x) {
		return x * x;
	}

	/// Calculates x^3.
	template <class T> inline T cube( const T & x) {
		return x * x * x;
	}
	
	///\brief Logistic function/logistic function.
	///
	///Calculates the sigmoid function 1/(1+exp(-x)). The type must be arithmetic. For example
	///float,double,long double, int,... but no custom Type. 
	template<class T>
	typename boost::enable_if<boost::is_arithmetic<T>, T>::type sigmoid(T x){
		if(x < minExpInput<T>()) {
			return 1;
		}
		if(x > maxExpInput<T>()) {
			return 0;
		}
		return 1. / (1.+ std::exp(-x));
	}
	
	///\brief Thresholded exp function, over- and underflow safe.
	///
	///Replaces the value of exp(x) for numerical reasons by the a threshold value if it gets too large.
	///Use it only, if there is no other way to get the function stable!
	///
	///@param x the exponent
	template<class T>
	T safeExp(T x ){
		if(x > maxExpInput<T>()){
			//std::cout<<"warning, x too high"<<std::endl;
			return 0.9 * std::numeric_limits<long double>::max();
		}
		//Allow Koenig Lookup
		using std::exp;
		return  exp(x);
	}
	///\brief Thresholded log function, over- and underflow safe.
	///
	///Replaces the value of log(x) for numerical reasons by the a threshold value if it gets too low.
	///Use it only, if there is no other way to get the function stable!
	///@param x the exponent
	template<class T>
	T safeLog(T x)
	{
		
		if(x> -std::numeric_limits<T>::epsilon()  && x < std::numeric_limits<T>::epsilon()){
			//std::cout<<"warning, x too low"<<std::endl;
			return boost::math::sign(x)*std::numeric_limits<T>::min_exponent;
		}
		
		//Allow Koenig Lookup
		using std::log;
		return log(x);
	};
	
	///\brief Numerically stable version of the function log(1+exp(x)).
	///
	///Numerically stable version of the function log(1+exp(x)).
	///This function is the integral of the famous sigmoid function.
	///The type must be arithmetic. For example
	///float,double,long double, int,... but no custom Type. 
	template<class T>
	typename boost::enable_if<boost::is_arithmetic<T>, T>::type softPlus(T x){
		if(x > maxExpInput<T>()){
			return x;
		}
		if(x < minExpInput<T>()){
			return 0;
		}
		return std::log(1+std::exp(x));
	}
	
	///\brief Numerically stable version of the function log(1+exp(x)). calculated with float precision to save some time
	///
	///Numerically stable version of the function log(1+exp(x)).
	///This function is the integral of the famous sigmoid function.
	inline double softPlus(double x){
		if(x > 15){
			return x;
		}
		if(x < -17){
			return 0;
		}
		return std::log(1.0f+std::exp(float(x)));
	}
	
	///brief lets x have the same sign as y.
	///
	///This is the famous well known copysign function from fortran.
	template<class T>
	T copySign(T x, T y){
		using std::abs;
		if(y > 0){
			return abs(x);
		}
		return -abs(x);
	}
	
	/// \brief Returns the trigamma function
	double trigamma(double x){
		if(x < 0)
		{
			//use reflection relation: -psi^1(1-x)-psi^1(x)=pi* d/dz cot(pi*x)
			//=> psi^1(x) = -psi^1(1-x)-pi* d/dz cot(pi*x)
			double pi = boost::math::constants::pi<double>();
			using std::sin;
			double s = sin(pi*x);
			double cot_pi_x=-pi / (s * s);
			return -trigamma(1-x) - pi * cot_pi_x;
		}
		//~ if(x > 20)
		//~ {
			//~ //reduce by multiplication theorem
			//~ //psi^1(2x)=1/4[psi^1(x)+psi^1(x+1/2]]
			//~ return 0.25*trigamma(x)+0.25*trigamma(x+1/2);
		//~ }
		//~ else{
		double g = 7;
		double p[] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028,
				771.32342877765313, -176.61502916214059, 12.507343278686905,
				-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
		double z = x-1;		
		double sum = 0;
		double sumSq = 0;
		double sumCube  = 0;
		for(std::size_t i = 1; i != 9; ++i){
			double zplusi=z+i;
			sum+=p[i]/zplusi;
			sumSq+=p[i]/sqr(zplusi);
			sumCube+=p[i]/cube(zplusi);
		}
		
		double t1=sumSq/(p[0]+sum);
		t1*=t1;
		double t2 = sumCube/(p[0]+sum);
		return 1/(z+g+0.5)+g/sqr(z+g+0.5)-t1+2*t2;
	}

}

/** @}*/ 

#endif 
