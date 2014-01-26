/**
*
* \brief Mamdami inference machine
* 
* \authors Marc Nunkesser
*/
#ifndef MAMDANIINFERENCEMACHINE_H
#define MAMDANIINFERENCEMACHINE_H


// include files:


#include <shark/Fuzzy/InferenceMachine.h>

namespace shark {
/**
 * \brief Mamdami inference machine
 *
 * A Mamdami inference is given by the fuzzy implication
 * \f$I(x,y)= min(x,y)\f$
 */
class MamdaniIM: public InferenceMachine {
public:
    /**
  * \brief Constructor
  *
  * @param rb the associated rule base
  */
    MamdaniIM( const boost::shared_ptr< RuleBase > & rb ) : InferenceMachine( rb ) {
    }

    /**
  * \brief Destructor
  */
    ~MamdaniIM() {}

private:
    OutputType buildTreeFast(
        RuleBase::rule_set_iterator & actual,
        unsigned int remainingRules,
        int conclusionNumber,
        const InputType in ) const {


        boost::shared_ptr<Rule> actualRule = *actual;
        const Rule::conclusion_type & actualConclusion = actualRule->conclusion();
        OutputType conclusionFragment( conclusionNumber );
        OutputType minNode( conclusionNumber );
        OutputType maxNode( conclusionNumber );
        boost::shared_ptr<ConstantFS> betaNode( new ConstantFS( actualRule->activation(in) ) );

        OutputType::iterator itf = conclusionFragment.begin();
        OutputType::iterator itMin = minNode.begin();
        for( Rule::conclusion_type::const_iterator it = actualConclusion.begin();
             it != actualConclusion.end();
             ++it, ++itf, ++itMin ) {
            *itf = *it;
            *itMin = Operators::min( betaNode, *itf );
        }

        if (remainingRules==0) {
            for (int i = 0;i<conclusionNumber;i++) minNode[i]->scale(actualRule->weight());
            return (minNode);
        } else {
            for (int i = 0;i<conclusionNumber;i++)  {
                minNode[i]->scale(actualRule->weight());
                maxNode[i].reset( new ComposedFS(ComposedFS::MAX,minNode[i],(buildTreeFast(++actual,--remainingRules,conclusionNumber,in))[i]) );
            };
            return(maxNode);
        };

    }



    //Build the tree of FuzzySets, whose evaluation yields the result of the c-th conclusion, starting with the a-th rule, and a ruleBase of length b.


};
}


#endif
