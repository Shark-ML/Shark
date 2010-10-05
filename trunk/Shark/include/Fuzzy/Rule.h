/**
 * \file Rule.h
 *
 * \brief  A rule which is composed of premise and conclusion
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

#ifndef RULE_H
#define RULE_H

#include <vector>
#include "RCPtr.h"
#include <stdarg.h>  //variable length parameter lists
#include "FuzzySet.h"
#include "FuzzyException.h"
#include "RCPtr.h"
#include "RCObject.h"



class FuzzyException;
class LinguisticTerm;
class RuleBase;
class NDimFS;

enum Connective {
	AND, 
	OR, 
	PROD, 
	PROBOR
}; //order is important cf parsing



/**
 * \brief  A rule which is composed of premise and conclusion.
 * 
 * The premise is composed of linguistic terms which are linked with one connective
 * (AND, OR, PROD or PROBOR). AND is the minimum function, OR the maximum function,
 * PROD the product, and PROBOR(x,y)= x+y-xy. There is always used solely
 * one kind of connective for all connections in one permise.
 * 
 *  
 */
class Rule: virtual public RCObject {
public:
	typedef std::vector< RCPtr<FuzzySet> >  IORuleType;
	typedef std::vector< RCPtr<FuzzySet> >  ConclusionType;


/**
 * \brief Default constructor
 *
 */
	Rule(Connective c=AND,
	     RuleBase * belongsTo = NULL,
	     double  _weight = 1.0);

/** 
 * \brief Constructor
 *  
 * Constructor for rules given by a string of the form
 *<pre>  IF LinguisticVariable1 = LinguisticTermA AND/OR
 *     LinguisticVariable2 = LinguisticTermB AND/OR ...
 *  THEN
 *     LinguisticVariable3 = LinguisticTermC
 *     LinguisticVariable4 = LinguisticTermD ...</pre>
 *
 *  where LinguisticVariable/Term in this example stands for their names,
 *  which can be obtained by their getName() methods.
 * 
 *  @param ruleText the rule given by a string
 *  @param belongsTo the RuleBase the rule belongs to
 *  @param _weight the weight of a rule 
 */
Rule( std::string ruleText,
	  RuleBase * belongsTo,
	  double _weight = 1.0 );

// destructor
virtual                      ~Rule();

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
double                       Activation(const std::vector<double> & inputs) const;




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
inline double                Activation(double input) const;
   
   
   

//Allow variable length parameter lists, where the first parameter indictes the number of double arguments: #ARGUMENTS, DOUBLE, DOUBLE...

	// double                       Activation(...) const;


/**
 * \brief Sets the connective, that shell be used in the premise.
 * 
 * @param con the Connective (AND, OR, PROD, or PROBOR) to be unsed in the premise
 */
void                         setConnective(Connective con);

/**
 * \brief Returns the rule given by a string.
 * 
 * @return string, that gives the rule
 */
std::string                  printRule() const;
	
	
/**
 * \brief Adds a linguistic therm to the premise.
 * 
 * @param lt the linguisic Term to be added
 */
virtual void                 addPremise(const RCPtr<LinguisticTerm> & lt);
	
/**
 * \brief Adds two linguistic therms to the premise.
 * 
 * @param lt1 the first linguisic term to be added
 * @param lt2 the secound linguistic term to be added
 */	
virtual void addPremise(const RCPtr<LinguisticTerm> & lt1,
	                    const RCPtr<LinguisticTerm> & lt2);
	                                        
/**
 * \brief Adds three linguistic therms to the premise.
 * 
 * @param lt1 the first linguisic term to be added
 * @param lt2 the secound linguistic term to be added
 * @param lt3 the third linguistic term to be added
 */		                                        
virtual void addPremise(const RCPtr<LinguisticTerm> & lt1,
	         const RCPtr<LinguisticTerm> & lt2,
	         const RCPtr<LinguisticTerm> & lt3);
	                                        
/**
 * \brief Adds four linguistic therms to the premise.
 * 
 * @param lt1 the first linguisic term to be added
 * @param lt2 the secound linguistic term to be added
 * @param lt3 the third linguistic term to be added
 * @param lt4 the fourth linguistic term to be added
 */	                                        
virtual void addPremise(const RCPtr<LinguisticTerm> & lt1,
	         const RCPtr<LinguisticTerm> & lt2,
	         const RCPtr<LinguisticTerm> & lt3,
	         const RCPtr<LinguisticTerm> & lt4);

// We are presuming complete conclusions, i.e. each rule in a rule base has the
// same number of entries in the conclusions and the n-th entry in each rule
// refers to the same output.

/**
 * \brief Adds a linguistic therm to the conclusion.
 * 
 * @param lt the linguisic Term to be added
 */
virtual void addConclusion(const  RCPtr<LinguisticTerm>& lt);

/**
 * \brief Returns the premise of a rule.
 * 
 * @return the premise of the rule
 */
NDimFS * getPremise();
	
/**
 * \brief Returns the conclusion of a rule. 
 * 
 * @return the conclusion of he rule
 */	
inline const ConclusionType& getConclusion() const {
	return(conclusion);
};


/**
 * \brief Returns the weight of a rule. 
 * 
 * @return the weight of he rule
 */	
inline double getWeight() const {
	return(weight);
};
	
protected:
	
	RuleBase *                   ruleBasePtr;
	
private:
	
	typedef double ConnectiveFuncType( double, double );
	
	ConnectiveFuncType*          connectiveFunc;

//Attributes:
	Connective                   ruleConnective;
	IORuleType                   premise;
	ConclusionType               conclusion;
	double                       weight;

//Methods:
	void                         initializePremise();
	// The following method is dangerous, because it allows to set the premise
	// which must be of normalized form, which is not checked. (cf addPremise)
	void                         setRule(IORuleType &, Connective, ConclusionType &);
	char*                        nextToken(char* tokenPtr);
};

#endif
