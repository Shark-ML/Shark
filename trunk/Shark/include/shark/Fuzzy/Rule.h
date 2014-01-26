/**
*
* \brief  A rule which is composed of premise and conclusion
* 
* \authors Marc Nunkesser
*/

/* $log$ */

#ifndef SHARK_FUZZY_RULE_H
#define SHARK_FUZZY_RULE_H

#include <shark/Fuzzy/Fuzzy.h>
#include <shark/Fuzzy/FuzzySet.h>

#include <shark/Fuzzy/LinguisticVariable.h>
#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/Operators.h>
// #include <shark/Fuzzy/FuzzySets/NDimFS.h>
#include <shark/Fuzzy/FuzzySets/HomogenousNDimFS.h>
#include <shark/Fuzzy/LinguisticTerms/ComposedLT.h>

#include <shark/LinAlg/Base.h>

#include <boost/shared_ptr.hpp>

#include <set>

namespace shark {

/**
 * \brief  A rule which is composed of premise and conclusion.
 *
 * The premise is composed of linguistic terms which are linked with one connective
 * (AND, OR, PROD or PROBOR). AND is the minimum function, OR the maximum function,
 * PROD the product, and PROBOR(x,y)= x+y-xy. There is always used solely
 * one kind of connective for all connections in one premise.
 *
 *
 */
class Rule {
public:
    typedef std::list< boost::shared_ptr<FuzzySet> > premise_type;
    typedef std::set< boost::shared_ptr< FuzzySet > >  conclusion_type;

    /**
            * \brief Default constructor
            *
            */
    Rule( Connective c = AND, double weight = 1.0 ) : m_weight( weight ) {
        setConnective( c );
    }

    // destructor
    virtual ~Rule() {}

    /**
  * \brief Returns the activation of the premise of the rule for a given input vector.
  *
  * This method returns the activation of the premise of the rule.
  * In this case the methods accepts crisp inputs.
  * Thus it is sufficient to calculate the value of the MF at the input points.
  * Input is a vector of singletons
  *
  * @param inputs input vector with values of the type double
  */
    double activation( const RealVector & inputs ) const {

        double out = 0;

        premise_type::const_iterator it = m_premise.begin();
        RealVector::const_iterator iti = inputs.begin();
        for( ; iti != inputs.end() && it != m_premise.end(); ++iti, ++it ) {

            if( *it ) {
                out = it == m_premise.begin() ? (**it)( *iti ) : mep_connectiveFunc( out, (**it)( *iti ) );
            }
        }

        return( out );
    }

    //If single input is required allow double as parameter: (instead of a vector of one element)

    /**
  * \brief Returns the activation of the premise of the rule for a given single imput value.
  *
  * This method returns the activation of the premise of the rule.
  * In this case the methods accepts crisp inputs.
  * Thus it is sufficient to calculate the value of the MF at the input points.
  * Input is a vector of singletons
  *
  * @param input single input value
  */
    inline double activation( double input ) const {
        return( activation( RealVector( 1, input ) ) );
    }

    /**
  * \brief Sets the connective, that shell be used in the premise.
  *
  * @param con the Connective (AND, OR, PROD, or PROBOR) to be unsed in the premise
  */
    void setConnective( Connective con ) {
        m_ruleConnective = con;

        switch( m_ruleConnective ) {
        case AND : //  the embodyment of pure evil!
            mep_connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::minimum);
            break;
        case OR :
            mep_connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::maximum);
            break;
        case PROD:
            mep_connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::prod);
            break;
        case PROBOR:
            mep_connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::probor);
            break;
        }

    }

    /**
  * \brief Returns the rule given by a string.
  *
  * @return string, that gives the rule
  */
    std::string print() const {
        std::stringstream ss;

        ss << "IF ";

        premise_type::const_iterator it = m_premise.begin(), itE = m_premise.end();
        while( it != itE ) {
            ss <<dynamic_cast<LinguisticTerm&>(**it).name() << " " << connective_to_name( m_ruleConnective ) << " ";
            ++it;
        }

        ss << "THEN ";

        conclusion_type::const_iterator itc = m_conclusion.begin(), itcE = m_conclusion.end();
        while( itc != itcE ) {
            ss << dynamic_cast<LinguisticTerm&>(**itc).name() << " ";
            ++itc;
        }

        return( ss.str() );

    }

    Connective connective() const {
        return( m_ruleConnective );
    }

    // We are presuming complete conclusions, i.e. each rule in a rule base has the
    // same number of entries in the conclusions and the n-th entry in each rule
    // refers to the same output.

    /**
  * \brief Adds a linguistic term to the conclusion.
  *
  * @param lt the linguistic Term to be added
  */
    virtual void addConclusion( const boost::shared_ptr<LinguisticTerm> & lt ) {
        m_conclusion.insert( lt );
    }

  /**
  * \brief Adds a linguistic term to the premise.
  *
  * @param lt the linguistic Term to be added
  */
    virtual void addPremise( const boost::shared_ptr<LinguisticTerm> & lt ) {
        m_premise.push_back( lt );;
    }




    /**
  * \brief Returns the premise of a rule, allows for l-value semantics.
  *
  * @return the premise of the rule
  */
    premise_type & premise() {
        return( m_premise );
    }

    /**
  * \brief Returns the premise of a rule.
  *
  * @return the premise of the rule
  */
    const premise_type & premise() const {
        return( m_premise );
    }

    /**
  * \brief Returns the conclusion of a rule.
  *
  * @return the conclusion of he rule
  */
    inline const conclusion_type & conclusion() const {
        return( m_conclusion );
    };


    /**
  * \brief Returns the weight of a rule.
  *
  * @return the weight of he rule
  */
    inline double weight() const {
        return( m_weight );
    };

protected:

    friend class RuleBase;

    typedef double ConnectiveFuncType( double, double );
    ConnectiveFuncType * mep_connectiveFunc;

    //Attributes:
    Connective m_ruleConnective;
    premise_type m_premise;
    conclusion_type m_conclusion;
    double m_weight;

    //Methods:
    void initializePremise();

    // The following method is dangerous, because it allows to set the premise
    // which must be of normalized form, which is not checked. (cf addPremise)
    void setRule(premise_type &, Connective, conclusion_type &);
};
}
#endif
