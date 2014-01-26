/**
*
* \brief A composite of rules
* 
* \authors Marc Nunkesser
*/

/* $log$ */

#ifndef SHARK_FUZZY_RULEBASE_H
#define SHARK_FUZZY_RULEBASE_H

#include <shark/Fuzzy/Rule.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <boost/shared_ptr.hpp>

#include <list>
#include <set>
#include <vector>

namespace shark {

	/**
	* \brief A composite of rules.
	* 
	* A rulebase combines several rules.  
	*/
	class RuleBase {
	public:
		typedef std::set< boost::shared_ptr<LinguisticVariable> > input_type;
		typedef input_type::iterator input_type_iterator;
		typedef input_type::const_iterator const_input_type_iterator;

		typedef std::set< boost::shared_ptr<LinguisticVariable> > output_type;
		typedef output_type::iterator output_type_iterator;
		typedef output_type::const_iterator const_output_type_iterator;

		typedef std::set< boost::shared_ptr<Rule> > rule_set_type;
		typedef rule_set_type::iterator rule_set_iterator;
		typedef rule_set_type::const_iterator const_rule_set_iterator;

		// this method inserts a new rule into the base and returns it,
		// so that the user can set it.
		/**
		* \brief Add a rule to the rulebase.
		* @param rule the rule to be added
		*/
		rule_set_iterator addRule( boost::shared_ptr<Rule> rule = boost::shared_ptr< Rule >() ) {
			if( !rule ) {
				rule.reset( new Rule() );
				rule->m_premise = Rule::premise_type( numberOfInputs(), boost::shared_ptr< FuzzySet >() );
			}
			
			std::pair< rule_set_iterator, bool > result = m_rules.insert( rule );
			return( result.first );
		}

		/**
		* \brief Adds a premise to the supplied rule.
		*
		* The general idea is to iterate over the inputFormat list of the
		* associated Rule Base and the premise in parallel in order to find
		* the "right place" for the input.
		* This means that we want the premise to be in a normalized form:
		* The order of the LTs must correspond to the order given by
		* inputFormat in the Rule Base.
		* Since there can be more than one LT of the same type, which is
		* incompatible with this normalized form, we merge these
		* terms to one sole LT using a ComposedLT.
		* Thus the precondition for addPremise is:
		*  - the premise is already in normalized form, containing null pointers
		*   where there is no LT of the demanded type.
		* Postcondition:
		*  - the premise is still in normalized form, containing now the
		*   additional LT given by the parameter lt.
		*
		* \param [in] rule Iterator pointing to the rule to be altered
		* \param [in] lt The linguistic term to be added.
		*/
		void addPremiseToRule( rule_set_iterator rule, const boost::shared_ptr< LinguisticTerm > & lt ) {

			(*rule)->premise().push_back( lt );

			//bool completed = false;
			//const_input_type_iterator formatIt = formatBegin();
			//Rule::premise_type::iterator premiseIt;
			
			/*if( (*rule)->premise().empty() ) {
				(*rule)->premise().push_back( lt );
				return;
			}*/

			/*for( premiseIt = (*rule)->premise().begin(); premiseIt != (*rule)->premise().end(); ++formatIt, ++premiseIt )
				//does the type of the lt to add match the type of the actual place in the input format?
			{
				if( lt->linguisticVariable() == (*formatIt) ) {
					completed = true; // the FS will now be added to the premise
					if (!(*premiseIt)) { // is this the first lt of this type in the premise?
						*premiseIt = lt;
					} else {
						// there is already a LT of the same type. Construct a ComposedLT out
						// of these the existing and the new one.
						switch( (*rule)-> connective() ) {
	case AND:
		*premiseIt = boost::shared_ptr<ComposedLT>(new ComposedLT(lt->name(),
			(lt->linguisticVariable()),
			ComposedFS::MIN,
			lt,
			*premiseIt));  //premiseIt
		break;
	case OR:
		*premiseIt = boost::shared_ptr<ComposedLT>(new ComposedLT(lt->name(),
			(lt->linguisticVariable()),
			ComposedFS::MAX,
			lt,
			*premiseIt));
		break;
	case PROD:
		*premiseIt = boost::shared_ptr<ComposedLT>(new ComposedLT(lt->name(),
			(lt->linguisticVariable()),
			ComposedFS::PROD,
			lt,
			*premiseIt));
		break;
	case PROBOR:
		*premiseIt = boost::shared_ptr<ComposedLT>(new ComposedLT(lt->name(),
			(lt->linguisticVariable()),
			ComposedFS::PROD,
			lt,
			*premiseIt));
		break;
	default:
		throw( shark::Exception( "Unknown Connective in rule.", __FILE__, __LINE__ ) );
		break;
						}
					}
				}
			}
			if (!completed) {
				throw( shark::Exception( "The type of the Linguistic Term does not match any type in the input format of the rule base", __FILE__, __LINE__ ) );
			}*/
		}

		/**
		* \brief Remove a rule from the rulbase.
		* @param rule the rule to be added
		*/	
		void removeRule( const boost::shared_ptr<Rule> & rule ) {
			m_rules.erase( rule );
		}

		/**
		* \brief Return the number of rules in the rulebase.
		* 
		*/
		inline std::size_t numberOfRules() const {
			return( m_rules.size() );
		}

		/**
		*\brief Return a string with the all the rules of the rulabase.
		* @return, the rulebase as a string
		*/
		std::string print() const {
			std::stringstream ss;

			for( const_rule_set_iterator it = ruleSetBegin();
				it != ruleSetEnd();
				++it ) {
				ss << (*it)->print() << std::endl;
			}

			return( ss.str() );
		}

		inline rule_set_iterator ruleSetBegin() {
			return( m_rules.begin() );
		};

		inline rule_set_iterator ruleSetEnd() {
			return( m_rules.end() );
		};

		inline const_rule_set_iterator ruleSetBegin() const {
			return( m_rules.begin() );
		};

		inline const_rule_set_iterator ruleSetEnd() const {
			return( m_rules.end() );
		};

		inline input_type_iterator formatBegin() {
			return( m_inputFormat.begin() );
		};

		inline input_type_iterator formatEnd() {
			return( m_inputFormat.end() );
		};

		inline const_input_type_iterator formatBegin() const {
			return( m_inputFormat.begin() );
		};

		inline const_input_type_iterator formatEnd() const {
			return( m_inputFormat.end() );
		};

		inline output_type_iterator conclusionsBegin() {
			return( m_outputFormat.begin() );
		};

		inline output_type_iterator conclusionsEnd() {
			return( m_outputFormat.end() );
		};

		inline const_output_type_iterator conclusionsBegin() const {
			return( m_outputFormat.begin() );
		};

		inline const_output_type_iterator conclusionsEnd() const {
			return( m_outputFormat.end() );
		};

		// the Input Format describes how a vector like (1,2,4,3) given as an input
		// to the rule must be interpreted, i.e. to which Linguistic Variables the
		// values refer. Thus the input format consists of a list of Linguistic-Variables

		/**
		* \brief Sets the Input Format.
		* @param in the input_type (list of linguistic variables)
		*/
		void setInputFormat( const input_type & in ) {
			m_inputFormat = in;
		}

		/**
		* \brief Adds up to four linguistic variable to the input format. 
		* 
		* The linguistic variables must be added in the correct order with respect to the
		* corresponding positions in the input vector.
		* 
		* @param lv1 the first linguistic variable to be added
		* @param lv2 the second linguistic variable to be added
		* @param lv3 the third linguistic variable to be added
		* @param lv4 the fourth linguistic variable to be added
		*/        												
		void addToInputFormat(const boost::shared_ptr<LinguisticVariable> & lv1,
			const boost::shared_ptr<LinguisticVariable> & lv2 = boost::shared_ptr< LinguisticVariable >(),
			const boost::shared_ptr<LinguisticVariable> & lv3 = boost::shared_ptr< LinguisticVariable >(),
			const boost::shared_ptr<LinguisticVariable> & lv4 = boost::shared_ptr< LinguisticVariable >()
			) {

				if( lv1 )
					m_inputFormat.insert( lv1 );
				if( lv2 )
					m_inputFormat.insert( lv2 );
				if( lv3 )
					m_inputFormat.insert( lv3 );
				if( lv4 )
					m_inputFormat.insert( lv4 );
		}


		/**
		* \brief Adds up to four linguistic variables to the output format. 
		* 
		* The linguistic variables must be added in the correct order with respect to the
		* corresponding positions in the input vector.
		* 
		* @param lv1 the firs linguistic variable to be added
		* @param lv2 the second linguistic variable to be added
		* @param lv3 the third linguistic variable to be added
		* @param lv4 the fourth linguistic variable to be added
		*/      
		void addToOutputFormat( const boost::shared_ptr<LinguisticVariable> & lv1,
			const boost::shared_ptr<LinguisticVariable> & lv2 = boost::shared_ptr< LinguisticVariable >(),
			const boost::shared_ptr<LinguisticVariable> & lv3 = boost::shared_ptr< LinguisticVariable >(),
			const boost::shared_ptr<LinguisticVariable> & lv4 = boost::shared_ptr< LinguisticVariable >() 
			) {

				if( lv1 )
					m_outputFormat.insert( lv1 );
				
				if( lv2 )
					m_outputFormat.insert( lv2 );

				if( lv3 )
					m_outputFormat.insert( lv3 );

				if( lv4 )
					m_outputFormat.insert( lv4 );

		}

		/**
		* \brief Removes a liniguistic variable from the Imput Format. 
		* 
		* @param lv the linguistic variable to be removed
		*/    
		void removeFromInputFormat( const boost::shared_ptr<LinguisticVariable>& lv ) {
			m_inputFormat.erase( lv );
		}

		/**
		* \brief Returns the number of linguistic variables in the Input Format.
		* 
		* @ return, the number of linguistic variables in the Input Format 
		*/      
		inline std::size_t numberOfInputs() const {
			return( m_inputFormat.size() );
		};

	protected:
		rule_set_type m_rules;
		input_type m_inputFormat;
		output_type m_outputFormat;
	};

}

#endif // SHARK_FUZZY_RULEBASE_H




