
/**
 * \file NDimFS.h
 *
 * \brief Base class for n-dimensional fuzzy sets
 * 
 * \authors Marc Nunkesser
 */


/* $log$ */


#ifndef NDIMFS_H
#define NDIMFS_H

#include <Fuzzy/FuzzySet.h>
#include <vector>
#include <algorithm>
#include <Fuzzy/Rule.h>
#include <Fuzzy/RCPtr.h>
#include <Fuzzy/RCObject.h>

/**
 * \brief Base class for n-dimensional fuzzy sets.
 *
 * Virtual base class for the classes HomogenousNDimFS and ComposedNDimFS.
 * A n-dimensional fuzzy set is described by the corresponding membership
 * function:
 * 
 * \f[
 *    \mu : X_1 \times X_2 \times \ldots \times X_n \rightarrow [0,1]
 * \f]
 * \f[
 *    \mu( x_1, x_2, \ldots, x_n ) = f( \mu_1(x_1), \mu_2(x_2),\ldots,\mu_n(x_n) )
 * \f]
 */
class NDimFS:public RCObject {
public:

	typedef std::vector< RCPtr<FuzzySet> > FuzzyArrayType;
	//enum                      Connective {AND, OR}; //order is important

	/**
	 * \brief Default Constructor.
	 */
	NDimFS();
	
	/**
	 * \brief Constructor.
	 *
	 * @param fat an array of fuzzy sets.
	 */
	NDimFS( FuzzyArrayType & fat );

	/**
	 * \brief Constructor for conversion from class FuzzySet.
	 *
	 * @param f a fuzzy set.
	 */
	NDimFS(const RCPtr<FuzzySet>& f);

	virtual ~NDimFS(); ///< Destructor

/**
 * \brief Returns the i'th component of the fuzzy set (the fuzzy set \f$\mu_i\f$). 
 * 
 * @param i the index of the fuzzy set to be returned
 * @return the i'th component (i.e. the fuzzy set \f$\mu_i\f$)
 */
	inline const RCPtr<FuzzySet> operator[](int i) const{
		return components[i];
	};

	/**
	 * \brief The n-dimensional \f$\mu\f$-function.
	 * 
	 * @param v the vector of values \f$(x_1,\ldots,x_n)\f$
	 * @return the value of the membership fuction at \f$(x_1,\ldots,x_n)\f$
	 */
	virtual double operator()(const std::vector<double> & v ) const = 0;
	// Also a list of doubles is accepted as input for operator()
	// There is no checking whether the given list is too long!

	// double  operator()(double) const { return 7.0;}
	//virtualdouble operator()(...) const;
	// A conversion to a simple FuzzySet is possible if the dimension equals one.
	//                        operator FuzzySet&() const;

/**
 * \brief Returns the dimension of a n-dimensional fuzzy set.
 *
 * @return the dimension of the fuzzy set.
 */
	inline unsigned int getDimension() const{
		return(components.size());
	};



protected:
	FuzzyArrayType components;
};

///////////////////////////////////////////////////////////////////
/////// inlined methods
///////////////////////////////////////////////////////////////////

//const FuzzySet* NDimFS::firstFuzzySet() const
//   { return components[0];};


#endif
