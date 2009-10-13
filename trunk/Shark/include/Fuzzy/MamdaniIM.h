
/**
 * \file MamdaniIM.h
 *
 * \brief A Mamdami inference machine
 * 
 * \authors Marc Nunkesser
 */


#ifndef MAMDANIINFERENCEMACHINE_H
#define MAMDANIINFERENCEMACHINE_H


// include files:


#include <Fuzzy/InferenceMachine.h>

/**
 * \brief A Mamdami inference machine.
 *
 * A Mamdami inference is given by 
 * \f$I(x,y)= min(x,y)\f$
 */
class MamdaniIM: public InferenceMachine {
public:
/**
 * \brief Constructor.
 * 
 * @param rb the associated rule base
 */
	MamdaniIM( RuleBase * rb );

/**
 * \brief Destructor
 */
	virtual ~MamdaniIM();

private:
	OutputType             buildTreeFast
	(RuleBase::BaseIterator & actual,
	 unsigned int remainingRules,
	 int conclusionNumber,
	 const InputType in) const;



//Build the tree of FuzzySets, whose evaluation yields the result of the c-th conclusion, starting with the a-th rule, and a ruleBase of length b.


};


#endif
