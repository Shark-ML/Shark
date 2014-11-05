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
#include <shark/Core/Exception.h>

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
	
	/// \brief Calculates the trigamma function -the derivative of the digamma function
	///
	/// The trigamma function is defined as
	/// \f[\Psi^{(1)}(x)= \frac{\partial^2}{\partial ^2} \log \gamma(x)\f]
	///
	/// It has poles as x=-1,-2,-3,.. at which the function value is infinity. 
	/// For more information see http://mathworld.wolfram.com/TrigammaFunction.html
	///
	/// We calculate it in two different ways. If the argument is small a taylor expansion
	/// is used. otherwise the argument is transformed to a valu large enough such that
	/// the asymptotic series holds which is then easy to calculate with high precision
	inline double trigamma(double x){
		double small = 1e-4;//threshold for taylor expansion validity around 0
		double large = 15;//threshold above which the asymptotic formula is valid
		//coefficients for taylor expansion - value at 0
		double trigamma1 = 1.6449340668482264365; // pi^2/6 = Zeta(2) 
		double tetragamma1 = -2.404113806319188570799476;  // -2 Zeta(3) */
		//even Bernoulli coefficients B_2..B_10 for the asymptotic formula
		double b2 =  1./6;
		double b4 = -1./30;
		double b6 =  1./42;
		double b8 = -1./30;
		double b10 = 5./66;
		// singularities are at -1, -2, -3,...
		if((x <= 0) && (std::floor(x) == x)) {
			return -std::numeric_limits<double>::infinity();
		}
		//treating of general negative arguments
		if(x < 0)
		{
			//use reflection relation: -psi^1(1-x)-psi^1(x)=pi* d/dz cot(pi*x)
			//=> psi^1(x) = -psi^1(1-x)-pi* d/dz cot(pi*x)
			double pi = boost::math::constants::pi<double>();
			double s = std::sin(pi*x);
			double ddz_cot_pi_x=-pi / (s * s);
			return -trigamma(1-x) - pi * ddz_cot_pi_x;
		}
		// Use Taylor series if argument <= small
		//for this make use of the entity trigamma(x)=trigamma(x+1)+1/x^2
		//to move the origin to 1 and calculate the taylor expansion at 1 
		if(x <= small) {
			return 1/(x*x) + trigamma1 + tetragamma1*x;
		}
		
		//otherwise make use of the same entity
		//for n steps until x+n > large, which is the threshold for the
		// asymptotic formula to converge. 
		double result = 0;
		while(x < large) {
			result += 1/(x*x);
			x++;
		}
		//now apply the asymptotic formula
		//trigamma(x)=1/x+1/(2*x*x)+sum_k B_{2k}/z^{2k+1}
		if(x >= large) {
			double r = 1/(x*x);
			double t = (b4 + r*(b6 + r*(b8 + r*b10)));
			result += 0.5*r + (1 + r*(b2 + r*t))/x;
		}
		return result;
	}

	/// \brief Calculates the tetragamma function - the derivative of the trigamma function
	///
	/// The trigamma function is defined as
	/// \f[\Psi^{(2)}(x)= \frac{\partial}{\partial x}\Psi^{(1)}(x)= \frac{\partial^3}{\partial ^3} \log \gamma(x)\f]
	///
	/// The function is undefined for x=-1,-2,-3,... and an exception is thrown.
	/// For more information see http://mathworld.wolfram.com/PolygammaFunction.html
	///
	/// We calculate it in two different ways. If the argument is small a taylor expansion
	/// is used. otherwise the argument is transformed to a valu large enough such that
	/// the asymptotic series holds which is then easy to calculate with high precision
	inline double tetragamma(double x)
	{
		double small = 1e-4;//threshold foruse of taylor expansion if value is close to 0
		double large = 18;//threshold above which the asymptotic formula is valid
		//coefficients for taylor expansion at 1
		double tetragamma1 = -2.404113806319188570799476;  /* -2 Zeta(3) */
		double pentagamma1 = 6.49393940226682914909602217; /* 6 Zeta(4) */
		//even Bernoulli coefficients B_2..B_10 for the asymptotic formula
		double b2 =  1./6;
		double b4 = -1./30;
		double b6 =  1./42;
		double b8 = -1./30;
		double b10 = 5./66;
		// singularities are at -1, -2, -3,...
		if((x <= 0) && (std::floor(x) == x)) {
			throw SHARKEXCEPTION("[tetragamma] negative whole numbers ar enot allowed");
		}
		//treating of general negative arguments
		if(x < 0)
		{
			//use derivative of reflection relation of trigamma: 
			//-psi^2(1-x)-psi^2(x)=pi* d^2/dz^2 cot(pi*x)
			//=> psi^2(x) = -psi^2(1-x)-pi* d^2/dz^2 cot(pi*x)
			//and d^2/dz^2 cot(pi*x)= -pi d/dz 1/sin(pi*x)^2
			// = 2*pi^2 cos(pi*x)/sin(pi*x)^3
			double pi = boost::math::constants::pi<double>();
			double s_pi_x = std::sin(pi*x);
			double c_pi_x = std::cos(pi*x);
			double ddz2_cot_pi_x=2*sqr(pi)*c_pi_x/cube(s_pi_x);
			return -tetragamma(1-x) - pi * ddz2_cot_pi_x;
		}
		// Use Taylor series if argument <= small
		//for this make use of the entity tetragamma(x)=tetragamma(x+1)-2/x^3
		//to move the origin to 1 and calculate the taylor expansion at 1 
		if(x <= small) {
			return -2/cube(x) + tetragamma1 + pentagamma1*x;
		}
		
		//otherwise make use of the same entity
		//for n steps until x+n > large, which is the threshold for the
		// asymptotic formula to converge. 
		double result = 0;
		while(x < large) {
			result -= 2/cube(x);
			x++;
		}
		//now apply the asymptotic formula - the derivative of the asymptotic formula
		// of the trigamma function
		//tetragamma(x)=-1/x^2-1/(x^3)-sum_k (2k+1)*B_{2k}/z^{2(k+1)}
		
		if(x >= large) {
			double r = 1/(x*x);
			double t = (5*b4 + r*(7*b6 + r*(9*b8 + r*11*b10)));
			result -= r/x + r*(1 + r*(3*b2 + r*t));
		}
		return result;
		
	}

}
/** @}*/ 

#endif 
