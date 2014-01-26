/**
*
* \brief Sugeno inference machine
* 
* \authors Marc Nunkesser
*/

/* $log$ */


#ifndef SUGENOIM_H
#define SUGENOIM_H

#include <shark/Fuzzy/InferenceMachine.h>
#include <shark/Fuzzy/SugenoRule.h>
#include <shark/Fuzzy/FuzzySets/SingletonFS.h>

namespace shark {
	/**
	* \brief Sugeno inference machine
	*/
	class SugenoIM: public InferenceMachine {
	public:

		/**
		* \brief Constructor.
		* 
		* @param rb the associated rulebase
		*/
		SugenoIM( boost::shared_ptr<RuleBase> rb = boost::shared_ptr<RuleBase>() ) : InferenceMachine( rb ) {
		}

		/**
		* \brief Destructor
		*/
		virtual ~SugenoIM() {}

		/**
		* \brief Computes Sugeno inference
		*
		* @param input a vector of crisp values (an InputType)
		* @return the inference
		*/
		double computeSugenoInference( const InputType & input ) const {
			
			double nominator = 0;
			double denominator = 0;
			double temp;
			for( RuleBase::rule_set_iterator it = mep_ruleBase->ruleSetBegin(); it != mep_ruleBase->ruleSetEnd(); ++it ) {
				boost::shared_ptr<SugenoRule> sugenoRule = boost::dynamic_pointer_cast< SugenoRule >( *it );
				temp = sugenoRule->activation( input );
				// TODO: nominator += temp * sugenoRule->calculateConsequence(input);
				denominator += temp;
			}

			if (denominator!=0)
				return(nominator/denominator);
			else
				throw( shark::Exception("No rule was activated by the given input", __FILE__, __LINE__ ) );
		}

		/**
		* \brief Computes the Sugeno inference. 
		*
		* @param a the first crisp value
		* @param b the second crisp value
		* @return the inference
		*/
		double computeSugenoInference(double a, double b) const {
			shark::InferenceMachine::InputType v( 2 );
			v[0] = a;
			v[1] = b;

			return( computeSugenoInference( v ) );
		}

		/**
		* \brief Computes the Sugeno inference
		*
		* @param a the first crisp value
		* @param b the second crisp value
		* @param c the third crisp value
		* @return the inference
		*/
		double computeSugenoInference(double a, double b, double c) const {
			shark::InferenceMachine::InputType v( 3 );

			v[0] = a;
			v[1] = b;
			v[2] = c;

			return( computeSugenoInference( v ) );
		}

		/**
		* \brief Computes the Sugeno inference
		*
		* @param a the first crisp value
		* @param b the second crisp value
		* @param c the third crisp value
		* @param d the fourth crisp value
		* @return the inference
		*/
		double computeSugenoInference(double a, double b, double c, double d) const {
			shark::InferenceMachine::InputType v( 4 );

			v[0] = a;
			v[1] = b;
			v[2] = c;
			v[3] = c;

			return( computeSugenoInference( v ) );
		}

	protected:
		virtual OutputType buildTreeFast( RuleBase::rule_set_iterator & actual,
			unsigned int remainingRules,
			int conclusionNumber,
			const InputType in ) const {
				return( OutputType() );
		};
	};

}
#endif
