
/**
 * \file InferenceMachine.h
 *
 * \brief An inference machine
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef INFERENCEMACHINE_H
#define INFERENCEMACHINE_H

#include "RuleBase.h"
#include "FuzzySet.h"
#include "FuzzyException.h"
#include "RCPtr.h"

#include <iostream>
#include <fstream>
#include <vector>


/**
 * \brief An inference machine.
 *
 * A virtual basis class for the different inference machines.
 */
class InferenceMachine {
public:
	typedef std::vector< RCPtr<FuzzySet> > OutputType;
	typedef std::vector<double>    InputType;

/**
 * \brief Constructor.
 *
 * @param rb the associated rulebase.
 */
	InferenceMachine(RuleBase * const rb = NULL);

/**
 * \brief Destructor
 */
	virtual ~InferenceMachine();

/**
 * \brief Computes the inference. 
 *
 *  @param it a vector of crisp values (an InputType)
 *  @return a vector of fuzzy sets 
 */
	virtual OutputType     computeInference(const InputType it) const;

/**
 * \brief Computes the inference. 
 *
 *  @param a a crisp value
 *  @return a vector of fuzzy sets
 */
	virtual OutputType     computeInference(double a)  const;

/**
 * \brief Computes the inference. 
 *
 *  @param a first crisp value
 *  @param b second crisp value
 *  @return a vector of fuzzy sets
 */
	virtual OutputType     computeInference(double a, double b)  const;

/**
 * \brief Computes the inference. 
 *
 *  @param a first crisp value
 *  @param b second crisp value
 *  @param c thrid crisp value
 *  @return a vector of fuzzy sets
 */
	virtual OutputType     computeInference(double a, double b, double c)  const;

/**
 * \brief Computes the inference. 
 *
 *  @param a first crisp value
 *  @param b second crisp value
 *  @param c thrid crisp value
 *  @param d forth crisp value
 *  @return a vector of fuzzy sets
 */
	virtual OutputType     computeInference(double a, double b, double c, double d)  const;

/**
 * \brief Set the associated rule base.
 * 
 * @param rB the rule base
 */
	inline  void           setRuleBase(RuleBase * const rB);

/**
 * \brief Plots the defuzzification results for the whole input range into a 
 * gnuplot-suited file.
 * 
 * For evaluation the borders of the linguistic variables are taken into
 * consideration.
 *
 * @param fileName name of the file where the data of characteristic curve is saved
 * @param resolution the resolution of the characterestic curve  
 */
	void                   characteristicCurve( const std::string fileName = "curve.dat", 
												long int resolution = 50 ) const;
protected:
	RuleBase*              ruleBasePtr;
	virtual OutputType     buildTreeFast( RuleBase::BaseIterator & actual, unsigned int remainingRules, int conclusionNumber, const InputType in) const = 0;
private:

//Build the tree of FuzzySets, whose evaluation yields the result of the c-th conclusion, starting with the a-th rule, and a ruleBase of length b.

	virtual void             addToFile(double,std::ofstream &) const;
	virtual void             addToFile(double,double,std::ofstream &) const;
};

void InferenceMachine::setRuleBase(RuleBase * const rbp) {
	ruleBasePtr = rbp;
};
#endif
