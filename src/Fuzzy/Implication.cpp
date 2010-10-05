
/**
 * \file Implication.cpp
 *
 * \brief An implication
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

#include <math.h>
#include <Fuzzy/Implication.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/ConstantFS.h>
#include <Fuzzy/NDimFS.h>
#include <Fuzzy/ComposedNDimFS.h>
#include <Fuzzy/HomogenousNDimFS.h>


Implication::Implication(const RCPtr<NDimFS>& fsa,
                         const RCPtr<NDimFS>& fsb ,
                         ImplicationType imp):xfs(fsa),yfs(fsb) {
	switch (imp) {

	case ZADEH:
		usedFunction= Zadeh;
		break;
	case MAMDANI:
		usedFunction= Mamdani;
		break;
	case LUKASIEWICZ:
		usedFunction=Lukasiewicz;
		break;
	case GOEDEL:
		usedFunction= Goedel;
		break;
	case KLEENEDIENES:
		usedFunction= KleeneDienes;
		break;
	case GOGUEN:
		usedFunction= Goguen;
		break;
	case GAINESRESCHER:
		usedFunction= GainesRescher;
		break;
	case REICHENBACH:
		usedFunction= Reichenbach;
		break;
	case LARSEN:
		usedFunction= Larsen;
		break;
	};
};

Implication::~Implication() {};


double Implication::operator()(const std::vector<double>& x,const std::vector<double>& y) const {
	return(usedFunction((*xfs)(x),(*yfs)(y)));
};

RCPtr<ComposedNDimFS> Implication::operator()(const std::vector<double>& x, Lambda y) const {

	if (y!=Y) throw(FuzzyException(19,"Lambda::Y expected"));
	RCPtr<ConstantFS> constFS(new ConstantFS((*xfs)(x)));
	std::vector< RCPtr<FuzzySet> >* vec = new std::vector< RCPtr<FuzzySet> >;
	(*vec).push_back(constFS); // bad
	RCPtr<HomogenousNDimFS> ndfs(new HomogenousNDimFS(*vec));
	RCPtr<ComposedNDimFS> compFS(new ComposedNDimFS(ndfs,yfs,usedFunction));
	return(compFS);
};

RCPtr<ComposedNDimFS> Implication::operator()(Lambda x, const std::vector<double>& y) const {
//  if(x!=X) throw(FuzzyException(20,"Lambda::X expected"));
//   ConstantFS* constFS = new ConstantFS((*yfs)(y));
//   vector< FuzzySet * > * vec = new vector< FuzzySet * >;
//   (*vec).push_back(constFS);
//   HomogenousNDimFS* ndfs = new HomogenousNDimFS(*vec);
//   RCPtr<ComposedNDimFS> compFS = new ComposedNDimFS(xfs,ndfs,usedFunction);

	return(0);
};

double Implication::Zadeh(double x, double y) {
	return(std::max(std::min(x,y),1-x));
};

double Implication::Mamdani(double x, double y) {
	return(std::min(x,y));
};

double Implication::Lukasiewicz(double x, double y) {
	return(std::min(1.0,1.0-x+y));
};

double Implication::Goedel(double x, double y) {
	return((x<=y)?1.0:y);
};

double Implication::KleeneDienes(double x, double y) {
	return(std::max(1.0-x,y));
};

double Implication::Goguen(double x, double y) {
	return(fabs(x)<1E-8?1.0:std::min(1.0,y/x));
};

double Implication::GainesRescher(double x, double y) {
	return(x<=y?1.0:0.0);
};

double Implication::Reichenbach(double x, double y) {
	return(1.0-x+x*y);
};

double Implication::Larsen(double x, double y) {
	return(x*y);
};
