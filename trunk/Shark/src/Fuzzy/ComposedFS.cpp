/**
 * \file ComposedFS.cpp
 *
 * \brief A composed FuzzySet
 * 
 * \authors Marc Nunkesser
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

#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/FuzzyException.h>



ComposedFS::ComposedFS(Operator x ,const RCPtr<FuzzySet>& left,
                       const RCPtr<FuzzySet>& right,
                       double (*uF)(double,double) ):
		               op(x),
		               leftOperand(left),
		               rightOperand(right),
		               userDefinedOperator(uF) {};
		               
		               
ComposedFS::ComposedFS(Operator x,const RCPtr<FuzzySet>& left,
		               const RCPtr<FuzzySet>& right):
		               op(x), leftOperand(left),rightOperand(right) {};


ComposedFS::ComposedFS(const ComposedFS& rhs): // (deep) copy constructor
		op(rhs.op),
		leftOperand(rhs.leftOperand),
		rightOperand(rhs.rightOperand),
		userDefinedOperator(rhs.userDefinedOperator) {
	//rightOperand.freeze();
	//leftOperand.freeze();
}

double ComposedFS::mu( double x ) const {
	double l,r;
	switch (op) {
	case MIN:
		return(std::min((*leftOperand)(x),(*rightOperand)(x)));
		break;
	case MAX:
		return(std::max(leftOperand->operator()(x),rightOperand->operator()(x)));
		break;
	case PROD:
		return(leftOperand->operator()(x)*rightOperand->operator()(x));
		break;
	case PROBOR:
		l = leftOperand->operator()(x);
		r = rightOperand->operator()(x);
		return(l+r-l*r);
		break;
	case USER:
		return((*userDefinedOperator)((*leftOperand)(x),(*rightOperand)(x)));
		break;
	default:
		throw(FuzzyException(6,"Unknown member function type/operator"));
	};
}



double ComposedFS::getMin() const {
	double l,r;
	switch (op)  {
	case MIN:
		return(std::max(leftOperand->getMin(),rightOperand->getMin()));
		break;
	case MAX:
		return(std::min(leftOperand->getMin(),rightOperand->getMin()));
		break;
	case PROD:
		return(std::max(leftOperand->getMin(),rightOperand->getMin()));
	case PROBOR:
		l = leftOperand->getMin();
		r = rightOperand->getMin();
		return(std::min(l,r));
	case USER:
		return(std::min(leftOperand->getMin(),rightOperand->getMin()));
		// this is a simplification wich could lead to (slightly) incorrect
		// results depending on the user-defined operator
		break;
	default:
		throw(FuzzyException(6,"Unknown member function type/operator"))  ;
	};
};

double ComposedFS::getMax() const {
	double l,r;
	switch (op)  {
	case MIN:
		l= leftOperand->getMax();
		r=rightOperand->getMax();
		// cout<<"Linker Operand:"<<l<<endl;
		// cout<<"Rechter Operand:"<<r<<endl;
		return(std::min(l,r));
		break;
	case MAX:
		l=leftOperand->getMax();
		r=rightOperand->getMax();
		// cout<<"Linker Operand:"<<l<<endl;
		// cout<<"Rechter Operand:"<<r<<endl;
		return(std::max(l,r));
		break;
	case PROD:
		return(std::min(leftOperand->getMax(),rightOperand->getMax()));
	case PROBOR:
		l = leftOperand->getMax();
		r = rightOperand->getMax();
		return(std::max(l,r));
	case USER:
		l=leftOperand->getMax();
		r=rightOperand->getMax();
		// cout<<"Linker Operand:"<<l<<endl;
		// cout<<"Rechter Operand:"<<r<<endl;
		return(std::max(l,r));
		// this is a simplification wich could lead to (slightly) incorrect
		// results depending on the user-defined operator
		break;
	default:
		throw(FuzzyException(6,"Unknown member function type/operator"))  ;
	};
};
