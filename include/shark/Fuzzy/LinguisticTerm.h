/**
 *
 * \brief A single linguistic term
 * 
 * \author Marc Nunkesser
 */

/* $log$ */

#ifndef SHARK_FUZZY_LINGUISTICTERM_H
#define SHARK_FUZZY_LINGUISTICTERM_H

#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <string>

namespace shark {
// this class is part of a diamond inheritance, thus we use virtual inheritance
class LinguisticVariable; // forward declaration


/**
   * \brief A single linguistic term.
   * 
   * A linguistic term (like 'fast' or 'slow') is a named FuzzySet, which is assigned to a
   * LinguisticVariable (like 'speed'). <br> 
   * This is a virtual base class. The children are different kinds of linguistic
   * terms corresponding to the different kinds of fuzzy sets. 
   */
class LinguisticTerm : virtual public FuzzySet, public boost::enable_shared_from_this< LinguisticTerm > {
public:

    /**
     * \brief Creates a new LinguisticTerm.
     * 
     * @param name name of the linguistic term, e.g. "very fast"
     */
    LinguisticTerm(const std::string & name) : m_name( name ) {
    }

    /// Destructor
    virtual ~LinguisticTerm() {}

    /**
     * \brief Returns the name of the linguistic term.
     * @return the name
     */
    inline const std::string & name() const {
        return( m_name );
    };
    
    /**
     * \brief Sets the name of the linguistic term.
     * @param name the new name
     */
    void setName( const std::string & name ) {
        m_name = name;
    }

    /**
     * \brief Returns the LinguisticVariable the term is assigned to.
     */
    inline const boost::shared_ptr<LinguisticVariable> & linguisticVariable() const {
        return( m_parent );
    };
    
    /**
     * \brief Reassigns this term to a new LinguisticVariable.
     * @param lv the new linguistic variable
     */
    void setLinguisticVariable( const boost::shared_ptr<LinguisticVariable> & lv ) {
        m_parent = lv;
    }


    /**
     * \brief Defuzzification of the linguistic term accounting the bounds of the suppport 
     *  given by the corresponding linguistic variable. 
     *
     * @param errRel relative approximation error that is tollerated during numerical integration
     * @param recursionMax max. depth of recursion during numerical integration (i.e. max. \f$2^n\f$ steps)
     */
    double defuzzify( double errRel = FuzzySet::RELATIVE_ERROR(), int recursionMax = FuzzySet::RECURSION_MAX() ) const {
        return (FuzzySet::defuzzify(m_parent->lowerBound(), m_parent->upperBound(), errRel, FuzzySet::RECURSION_MAX()));
    };

protected:
    boost::shared_ptr<LinguisticVariable> m_parent; // Pointer to Linguistic Variable the term belongs to
    std::string m_name;

};
}
#endif
