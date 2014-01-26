/**
*
* \brief Operators and connective functions
* 
* \authors Marc Nunkesser
*/


/* $log$ */
#ifndef SHARK_FUZZY_OPERATORS_H
#define SHARK_FUZZY_OPERATORS_H

#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/Implication.h>

#include <shark/Fuzzy/FuzzySets/SingletonFS.h>
#include <shark/Fuzzy/FuzzySets/ComposedFS.h>

#include <boost/shared_ptr.hpp>

#include <vector>

namespace shark {
	/**
	* \brief Operators and connective functions.
	*
	* This class implements operators needed to do some calculations on fuzzy sets and the 
	* connective functions MINIMUM, MAXIMUM, PROD and PRODOR.	
	*/
	class Operators {
	public:

		/**
		* \brief Connects two fuzzy sets via the maximum function and returns the resulting fuzzy set.
		* 
		* @param f1 the first fuzzy set
		* @param f2 the second fuzzy set
		* @return the resulting composed fuzzy set
		*/
		static boost::shared_ptr<ComposedFS> max( const boost::shared_ptr<FuzzySet>& f1, const boost::shared_ptr<FuzzySet>& f2 ) {
			boost::shared_ptr< ComposedFS > cfs( new ComposedFS( ComposedFS::MAX, f1, f2 ) );
			return( cfs );
		}


		// inline static boost::shared_ptr<SingletonFS> min(const boost::shared_ptr<FuzzySet>& ,const boost::shared_ptr<SingletonFS>&);
		// inline static boost::shared_ptr<SingletonFS>  min(const boost::shared_ptr<SingletonFS>&,const boost::shared_ptr<FuzzySet>& );

		/**
		* \brief Connects two fuzzy sets via the minimum function and returns the resulting fuzzy set.
		* 
		* @param f1 the first fuzzy set
		* @param f2 the second fuzzy set
		* @return the resulting composed fuzzy set
		*/
		static boost::shared_ptr<FuzzySet> min( const boost::shared_ptr<FuzzySet>& f1, const boost::shared_ptr<FuzzySet>& f2 ) {
			if( ::fabs( f1->min() - f1->max() ) < 1E-8 )
				return( minLFS( f1, f2 ) );

			if( ::fabs( f2->min() - f2->max() ) < 1E8 )
				return( minLFS( f2, f1 ) );

			return( boost::shared_ptr< FuzzySet >( new ComposedFS( ComposedFS::MIN, f1, f2 ) ) );
		}
		// sup min composition for (vector of ) singleton input:
		// instead of singletons one must pass directly their
		// double values i.e. Singleton.defuzzify()


		/**
		* \brief Sup-min composition of an implication and a vector of singeltons.
		*
		* \f[
		*      \mu(y) = \sup_{x} min(\mu_1(x), \mu_2(x,y))
		* \f]
		* where \f$\mu_2(x,y)\f$ is the implication function
		* and \f$\mu_1(x)\f$ is the current input, in our case a 
		* vector of singletons, which simplifies this to
		*
		* \f[
		*      \mu(y) =\mu_2(singeltonInput, y)
		* \f]	
		*
		* @param input the input vector (vector of singeltons)
		* @param imp the implication
		* @return the resulting composed n-dimensional fuzzy set 
		*/
		template<typename Implication>
		static boost::shared_ptr<ComposedNDimFS> supMinComp( const RealVector & input, Implication * imp ) {
			if( imp == NULL ) 
				return( boost::shared_ptr< ComposedNDimFS >() );

			return( (*imp)( input, Implication::Y ) );
		}

		// Connective Functions

		/**
		* \brief The minimum function.
		* 
		* @param a first input value
		* @param b second input value
		* retrun the minimum of a and b
		*/
		inline static double minimum( double a, double b ) {
			return( std::min( a, b ) );
		};

		/**
		* \brief The maximum function.
		* 
		* @param a first input value
		* @param b second input value
		* retrun the maximum of a and b
		*/
		inline static double maximum( double a, double b ) {
			return( std::max( a, b ) );
		};

		/**
		* \brief The PROD function.
		* 
		* @param a first input value
		* @param b second input value
		* retrun the product of a and b (a*b)
		*/
		inline static double prod( double a, double b ) {
			return( a * b );
		};


		/**
		* \brief The PROBOR function.
		* 
		* @param a first input value
		* @param b second input value
		* retrun PROBOR(a,b)= a+b-a*b
		*/
		inline static double probor( double a, double b ) {
			return(a+b-a*b);
		};

	private:

		inline static  boost::shared_ptr<SingletonFS> minLFS(const boost::shared_ptr<FuzzySet> & s, const boost::shared_ptr<FuzzySet> & fs ) {
			return( boost::shared_ptr< SingletonFS >( new SingletonFS( s->defuzzify(), (*fs)( s->defuzzify() ) ) ) );
		}
	};
}
#endif
