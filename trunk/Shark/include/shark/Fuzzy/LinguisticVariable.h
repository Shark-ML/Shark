/**
*
* \brief A composite of linguistic terms
* 
* \authors Marc Nunkesser
*/

#ifndef LINGUISTICVARIABLE_H
#define LINGUISTICVARIABLE_H

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <string>
#include <map>
#include <limits>
#include <list>
#include <vector>

namespace shark {

class LinguisticTerm;

/**
 * \brief A composite of linguistic terms.
 *
 * A linguistic variable combines linguistic terms like 'slow', 'moderately fast',
 * 'fast', 'unbelievable fast' under a new name, e.g. 'speed'.
 */
class LinguisticVariable : public boost::enable_shared_from_this< LinguisticVariable > {
public:

    typedef std::vector< boost::shared_ptr< LinguisticTerm > >::iterator iterator;
    typedef std::vector< boost::shared_ptr< LinguisticTerm > >::const_iterator const_iterator;

    /**
  * \brief The constructor sets the name and the range for the new linguistic
  * variable, e.g. ( "Temperature", -10, 120 ).
  *
  * The bounds also hold for the support of all derived fuzzy sets, i.e.
  * regions outside these bounds are ignored for defuzzyfication.
  *
  * @param name the variables name, e.g. "Temperature"
  * @param lowerBound lower bound of the variables range
  * @param upperBound upper bound of the variables range
  */
    LinguisticVariable(	const std::string & name = "",
			double lowerBound = -std::numeric_limits<double>::max(),
			double upperBound = std::numeric_limits<double>::max()
			) : m_name( name ),
        m_lowerBound( lowerBound ),
        m_upperBound( upperBound ) {
    }

    /**
      * \brief Virtual d'tor.
      */
    virtual ~LinguisticVariable() {
    }

    /// Returns the name of the linguistic variable.
    const std::string & name() const {
        return( m_name );
    }

    /**
  * \brief Sets the name of the linguistic variable.
  * @param name the new name
  */
    void setName( const std::string & name ) {
        m_name = name;
    }

    /**
  * \brief Adds a new linguistic term.
  *
  * This method shouldn't be necessary most of time, since a LinguisticTerm
  * is associated with its LinguisticVariable durin construction.
  *
  * @param lt the LinguisticTerm to add
  */
    template< typename LinguisticTerm >
    void addTerm( const boost::shared_ptr< LinguisticTerm > & lt ) {

        // TV: Do we need to handle non-duplicate terms here?
        iterator it = std::find( begin(), end(), lt );
        if( it == end() ) {
            lt->setLinguisticVariable( shared_from_this() );
            m_terms.push_back( lt );
        }

    }

    /**
  * \brief Removes the given LinguisticTerm from this variable.
  * @param lt the LinguisticTerm to remove
  */
    void removeTerm( const boost::shared_ptr<LinguisticTerm> & lt ) {
        iterator it = std::find( begin(), end(), lt );
        if( it != end() )
            m_terms.erase( it );
    }

    /// Returns number of terms associated with the LinguisticVariable
    inline std::size_t numberOfTerms() const {
        return( m_terms.size() );
    };

    /**
  * \brief Returns a LinguisticTerm associated with this variable.
  * @param whichOne index of the term ( 0 <= index < numerOfTerms )
  */
    const boost::shared_ptr<LinguisticTerm> & getTerm( std::size_t whichOne) const {
        return( m_terms[ whichOne ] );
    }

    boost::shared_ptr<LinguisticTerm> & getTerm( std::size_t whichOne ) {
        return( m_terms[ whichOne] );
    }

    std::size_t size() const {
        return( m_terms.size() );
    }

    iterator begin() {
        return( m_terms.begin() );
    }

    iterator end() {
        return( m_terms.end() );
    }

    const_iterator begin() const {
        return( m_terms.begin() );
    }

    const_iterator end() const {
        return( m_terms.end() );
    }

    /**
  * \brief Sets a new range for the variable.
  *
  * The bounds also hold for the support of all derived fuzzy sets.
  *
  * @param lower the lower bound
  * @param upper the upper bound
  */
    void setBounds( double lower, double upper ) {
        m_lowerBound = lower;
        m_upperBound = upper;
    }

    /// Returns the lower bound of the variables range.
    inline double lowerBound() const {
        return( m_lowerBound );
    };

    /// Returns the upper bound of the variables range.
    inline double upperBound() const {
        return( m_upperBound );
    };

protected:
    typedef std::vector< boost::shared_ptr<LinguisticTerm> > Termset;

    Termset m_terms;
    std::string m_name;

    double m_lowerBound; //c.f constructor
    double m_upperBound;

    // typedef std::map< std::string, LinguisticVariable*, std::less<std::string> > mapType;
    // static mapType   lvMap;

};
}
#endif
