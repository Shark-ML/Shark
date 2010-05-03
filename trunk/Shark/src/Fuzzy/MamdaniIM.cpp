/**
 * \file MamdaniIM.cpp
 *
 * \brief A Mamdami inference machine
 * 
 * \authors Marc Nunkesser
 */


#include <Fuzzy/MamdaniIM.h>
#include <Fuzzy/Rule.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/ConstantFS.h>
#include <Fuzzy/Operators.h>


MamdaniIM::MamdaniIM(RuleBase * rbp): InferenceMachine(rbp)
{ }

MamdaniIM::~MamdaniIM()
{ }


MamdaniIM::OutputType MamdaniIM::buildTreeFast
(RuleBase::BaseIterator & actual,
 unsigned int remainingRules,
 int conclusionNumber,
 const  InputType in) const {

	RCPtr<Rule> actualRule = *actual;
	const Rule::ConclusionType& actualConclusion = actualRule->getConclusion();
	OutputType conclusionFragment(conclusionNumber);
	OutputType minNode(conclusionNumber);
	OutputType maxNode(conclusionNumber);
	RCPtr<ConstantFS> betaNode(new ConstantFS(actualRule->Activation(in)));

	for (int i = 0;i<conclusionNumber;i++) {
		conclusionFragment[i] = (actualConclusion)[i];
		minNode[i] = Operators::min(betaNode,conclusionFragment[i]);
	};

	if (remainingRules==0) {
		for (int i = 0;i<conclusionNumber;i++) minNode[i]->scale(actualRule->getWeight());
		return (minNode);
	} else {
		for (int i = 0;i<conclusionNumber;i++)  {
			minNode[i]->scale(actualRule->getWeight());
			maxNode[i]= new ComposedFS(ComposedFS::MAX,minNode[i],(buildTreeFast(++actual,--remainingRules,conclusionNumber,in))[i]);
		};
		return(maxNode);
	};
};
