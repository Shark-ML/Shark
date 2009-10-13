
/**
 * \file Operators.cpp
 *
 * \brief Operators and connective functions
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#include "Fuzzy/Operators.h"

RCPtr<ComposedFS> Operators::max(const RCPtr<FuzzySet>& l ,const RCPtr<FuzzySet>& r) {
	RCPtr<ComposedFS> cfs(new ComposedFS(ComposedFS::MAX,l,r));
	return(cfs);
};

/* RCPtr<SingletonFS> Operators::min(const RCPtr<FuzzySet>& fs,const RCPtr<SingletonFS>& s) */
/*  {  RCPtr<SingletonFS> sfs(new SingletonFS(s->defuzzify(),(*fs)(s->defuzzify()))); */
/*    // cout<<"Spezieller Konstruktor"<<endl; */
/*    return(sfs); */
/*  }; */

/* RCPtr<SingletonFS> Operators::min(const RCPtr<SingletonFS>& s,const RCPtr<FuzzySet>& fs) */
/* {  RCPtr<SingletonFS> sfs(new SingletonFS(s->defuzzify(),(*fs)(s->defuzzify()))); */
/*    return(sfs); */
/* };    */

RCPtr<SingletonFS> Operators::minLFS(const RCPtr<FuzzySet>&  s,const RCPtr<FuzzySet>& fs) {
	RCPtr<SingletonFS> sfs(new SingletonFS(s->defuzzify(),(*fs)(s->defuzzify())));
	return(sfs);
};

RCPtr<FuzzySet> Operators::min(const RCPtr<FuzzySet>& l, const RCPtr<FuzzySet>& r) {
	RCPtr<FuzzySet> result;
	if (fabs(l->getMin()-l->getMax())<1E-8)
		// is the left operand a singleton?
		// this simplification restricts numbers to numbers
		// whose significant digits begin before 1E-8
	{
		result = minLFS(l,r);
		return result;
	};
	if (fabs(r->getMin()-r->getMax())<1E-8)
		// is the right operand a singleton
	{
		result = minLFS(r,l);
		return result;
	};
	ComposedFS* cfs = new ComposedFS(ComposedFS::MIN,l,r);
	return(cfs);
};

RCPtr< ComposedNDimFS > Operators::supMinComp(const std::vector<double>  & d, Implication* imp) {
	// cf Bothe p126: mu(y)=sup_x min(mu_a(x),mu_m(x,y)
	// where mu_m(x,y) is the implication function
	// and mu_a is the current input, in our case a
	// vector of singletons, which simplifies this to
	// mu(y)=mu_m(singleton_input,y)
	// where our lambda feature yields a convenient solution
	RCPtr<ComposedNDimFS> ndfs = (*imp)(d,Implication::Y);
	return(ndfs);
};
