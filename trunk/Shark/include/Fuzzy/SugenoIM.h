
/**
 * \file SugenoIM.h
 *
 * \brief A Sugeno inference machine.
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


#ifndef SUGENOIM_H
#define SUGENOIM_H

#include <Fuzzy/InferenceMachine.h>
#include <Fuzzy/SingletonFS.h>

/**
 * \brief A Sugeno inference machine.
 */
class SugenoIM: public InferenceMachine {
public:


/**
 * \brief Constructor.
 * 
 * @param rb the associated rulebase
 */
	SugenoIM( RuleBase* rb = 0 );

/**
 * \brief Destructor
 */
	virtual ~SugenoIM();

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param inputType a vector of crisp values (an InputType)
 * @return the inference
 */
	double computeSugenoInference(const InputType inputType) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @param c the third crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b, double c) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @param c the third crisp value
 * @param d the fourth crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b, double c, double d) const;

private:

	virtual void             addToFile(double,std::ofstream &) const;
	virtual void             addToFile(double,double,std::ofstream &) const;
private:
	inline  virtual OutputType       buildTreeFast
	(RuleBase::BaseIterator & actual,
	 unsigned int remainingRules,
	 int conclusionNumber,
	 const InputType in) const {
		return( OutputType() );
	};
};

#endif
