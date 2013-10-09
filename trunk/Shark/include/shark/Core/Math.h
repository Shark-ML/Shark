/**
*
*  \brief Very basic math abstraction layer.
*
*  \par
*  This file serves as a minimal abstraction layer.
*  Inclusion of this file makes some frequently used
*  functions, constants, and header file inclusions
*  OS-, compiler-, and version-independent.
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
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

/** @}*/ 


};

#endif 
