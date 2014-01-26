/**
 *
 * \brief An user defined inference machine
 * 
 * \authors Marc Nunkesser
 */


#ifndef CUSTOMINFERENCEMACHINE_H
#define CUSTOMINFERENCEMACHINE_H


#include <shark/Fuzzy/InferenceMachine.h>
#include <shark/Fuzzy/Implication.h>

namespace shark {
/**
 * \brief An user defined inference machine.
 *
 * This class enables the user to configurate an inference by his own.
 * An inference here is given as a sup-min composition of implication and
 * premise. 
 * \f[
 *      \mu(y) = \sup_{x} min(\mu_1(x), \mu_2(x,y))
 * \f]
 * Where \f$\mu_1\f$ is the premise and \f$\mu_2\f$ is the implication function.
 * Thus the inference mechanism is influenced by the implication
 * choosen by the user.
 */
class CustomIM: public InferenceMachine {
public:

/**
 * \brief Constructor
 * 
 * @param rb the associated rulebase
 * @param im the type of implication
 */
	CustomIM(RuleBase * rb ,Implication::ImplicationType im);

/**
 * \brief Destructor
 */
	virtual ~CustomIM();
	
private:
	Implication::ImplicationType  usedImplication;
	OutputType buildTreeFast (	RuleBase::BaseIterator & actual,
	                           unsigned int remainingRules,
	                           int conclusionNumber,
	                           const InputType in) const;

};

}
#endif
