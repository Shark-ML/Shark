
/**
 * \file RuleBase.h
 *
 * \brief A composite of rules
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */


#ifndef RULEBASE_H
#define RULEBASE_H


#include "Rule.h"
#include <list>
#include "LinguisticVariable.h"
#include "RCPtr.h"
#include "Rule.h"



/**
 * \brief A composite of rules.
 * 
 * A rulebase combines several rules.  
 */
class RuleBase {

public:

	RuleBase();
	~RuleBase();

	typedef std::list < RCPtr<LinguisticVariable> > inputTemplate;
	typedef std::list < RCPtr<LinguisticVariable> > outputTemplate;

// this method inserts a new rule into the base and returns it,
// so that the user can set it.
    /**
	 * \brief Add a rule to the rulebase.
	 * @param rule the rule to be added
	 */
	void addRule(const RCPtr<Rule> & rule);
	
    /**
	 * \brief Remove a rule from the rulbase.
	 * @param rule the rule to be added
	 */	
	void removeRule(const RCPtr<Rule> & rule);
	
	/**
	 * \brief Return the number of rules in the rulebase.
	 * 
	 */
	inline int getNumberOfRules() {
		return(rules.size());
	};
	
	/**
	 *\brief Return a string with the all the rules of the rulabase.
	 * @return, the rulebase as a string
	 */
	const std::string print() const;
	
	/**
	 * \brief Return a certain rule of the rulebase. 
	 * @param whichone the number of the rule that shall be returned
	 */
	const RCPtr<Rule> getRule(int whichone);

private:
	typedef std::list < RCPtr<Rule> > RuleSet;
	
	
public:
	
	typedef RuleSet::const_iterator BaseIterator;
	
	inline BaseIterator getFirstIterator() const {
		return(rules.begin());
	};
	
	inline BaseIterator                    getLastIterator() const {
		return(rules.end());
	};
	
	typedef inputTemplate::const_iterator  FormatIterator;
	
	inline FormatIterator                  getFirstFormatIterator() const {
		return(inputFormat.begin());
	};
	
	inline FormatIterator                  getLastFormatIterator() const {
		return(inputFormat.end());
	};
	
	typedef outputTemplate::const_iterator ConclIt;
	
	inline FormatIterator                  getFirstConclIt() const {
		return(outputFormat.begin());
	};
	
	inline FormatIterator                  getLastConclIt() const {
		return(outputFormat.end());
	};
	
// the Input Format describes how a vector like (1,2,4,3) given as an input
// to the rule must be interpreted, i.e. to which Linguistic Variables the
// values refer. Thus the input format consists of a list of Linguistic-Variables

	/**
	 * \brief Sets the Input Format.
	 * @param in the InputTemplate (list of linguistic vaiables)
	 */
     void setInputFormat(inputTemplate & in);
	
	/**
	 * \brief Adds a liniguistic variable to the Imput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv the linguistic variable to be added
	 */
	void addToInputFormat( const RCPtr<LinguisticVariable>& lv );
	
	/**
	 * \brief Adds two liniguistic variables to the Imput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv1 the first linguistic variable to be added
	 * @param lv2 the secound linguistic variable to be added
	 */
	void addToInputFormat(const RCPtr<LinguisticVariable>& lv1,
	                      const RCPtr<LinguisticVariable>& lv2);
	            											
	/**
	 * \brief Adds three liniguistic variables to the Imput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv1 the first linguistic variable to be added
	 * @param lv2 the secound linguistic variable to be added
	 * @param lv3 the third linguitic variable to be added
	 */            													
	void addToInputFormat( const RCPtr<LinguisticVariable>& lv1,
						   const RCPtr<LinguisticVariable>& lv2,
						   const RCPtr<LinguisticVariable>& lv3);
	
	/**
	 * \brief Adds four liniguistic variable to the Imput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv1 the first linguistic variable to be added
	 * @param lv2 the secound linguistic variable to be added
	 * @param lv3 the third linguitic variable to be added
	 * @param lv4 the fourth linguitic variable to be added
	 */        												
	void addToInputFormat(const RCPtr<LinguisticVariable>& lv1,
	                      const RCPtr<LinguisticVariable>& lv2,
	                      const RCPtr<LinguisticVariable>& lv3,
	                      const RCPtr<LinguisticVariable>& lv4);
	        												
	/**
	 * \brief Adds a liniguistic variable to the Ouput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv the linguistic variable to be added
	 */        												
	void addToOutputFormat(const RCPtr<LinguisticVariable>& lv);
	
	/**
	 * \brief Adds two liniguistic variables to the Ouput Format. 
	 * 
	 * The linguistic variables must be added in the correct order with respect to the
	 * corresponding positions in the input vector.
	 * 
	 * @param lv1 the firs linguistic variable to be added
	 * @param lv2 the secound linguistic variable to be added
	 */      
	void addToOutputFormat( const RCPtr<LinguisticVariable>& lv1,
							const RCPtr<LinguisticVariable>& lv2 );
													        
	/**
	 * \brief Removes a liniguistic variable from the Imput Format. 
	 * 
	 * @param lv the linguistic variable to be removed
	 */    
	void removeFromInputFormat(const RCPtr<LinguisticVariable>& lv);
	
	/**
	 * \brief Returns the number of linguistic variables in the Input Format.
	 * 
	 * @ return, the number of linguistic variables in the Input Format 
	 */      
	inline unsigned int getNumberOfInputs() const {
		return(inputFormat.size());
	};

private:
	RuleSet                                rules;
	inputTemplate                          inputFormat;
	outputTemplate                         outputFormat;
	bool                                   destructorFlag;

};

#endif




