//===========================================================================
/*!
 *  \file UncertaintyQuantification.cpp
 *
 *  \brief Uncertainty quantification for rank-based selection
 *
 *  \author  T. Glasmachers, based on code by C. Igel
 *  \date    2008
 *
 * based on
 *
 * "A Method for Handling Uncertainty in Evolutionary Optimization
 * with an Application to Feedback Control in Combustion"
 * N. Hansen, A.S.P. Niederberger, L. Guzzella, and P. Koumoutsakos
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      EALib
 *
 *
 *
 *  This file is part of ReClaM. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *
*/
//===========================================================================



#include <iomanip>
#include <vector>
#include <EALib/UncertaintyQuantification.h>


// Excel way to compute percentiles
template<class T> double percentile(std::vector<T> v, double p =.25) {
	if (v.empty()) throw SHARKEXCEPTION("[percentile] list must not be empty");
	std::sort(v.begin(),v.end());
	unsigned N = v.size();

	double n = p * (double(N) - 1.0) + 1.0;
	unsigned k = unsigned(floor(n));
	double d = n - k;
	if(k == 0) return v[0];
	if(k == N) return v[N - 1];

	return v[k - 1] + d * (v[k] - v[k - 1]);
}


// data structure for reevaluation and re-ranking 
struct RankingObject {
	RankingObject(double x, unsigned i) : fitnessOld(x), fitnessNew(x), index(i), reev(false) {};
	double fitnessOld;
	double fitnessNew;
	unsigned rankNew;
	unsigned rankOld;
	unsigned Delta;
	unsigned index;
	bool reev;
};

bool rankingObjectLessFitnessOld(const RankingObject &x, const RankingObject &y) {
	if(x.fitnessOld < y.fitnessOld) return true;
	return false;
}
bool rankingObjectLessFitnessNew(const RankingObject &x, const RankingObject &y) {
	if(x.fitnessNew < y.fitnessNew) return true;
	return false;
}

// computes the limit rank change depending on theta (step 5)
double deltaLimTheta(unsigned R, unsigned l, double theta) {
	std::vector<double> v;
	// for a given rank R, the multi-set of all possible rank changes are computed
	for(unsigned i=1; i<(2*l-1); i++) v.push_back(fabs(double(i)-double(R)));
	// the theta/2 percentile of the multi-set is returned
	return percentile(v, 0.5 * theta);
}

// compute uncertainty level s
double UncertaintyQuantification(Population& p, NoisyFitnessFunction& f, double theta, double r_l) {
	unsigned l = p.size();
	unsigned i;

	// initialize data structure for reevaluation and re-ranking (step 1)
	std::vector<RankingObject> v;
	for (i=0; i<l; i++) 
		v.push_back(RankingObject(p[i].getFitness(), i));
	for (i=0; i<l; i++) 
		v.push_back(RankingObject(p[i].getFitness(), i));

	// compute the number of solutions to be reevaluated (step 2)
	if (!r_l) r_l = Shark::max(0.1, 2.0 / l);
	unsigned l_reev = unsigned(floor(r_l * l));
	if (Rng::coinToss(r_l * l - floor(r_l * l))) l_reev++;

	// reevaluate first l_reev individuals (step 3)
	// no perturbation is applied, does not work for frozen noise
	for (i=0; i<l_reev; i++) {
		v[i].fitnessNew = f.fitness(dynamic_cast< std::vector< double >& >(p[i][0]));
		v[i].reev = true;
	}

	// compute rank using original fitness
	std::sort(v.begin(), v.end(), &rankingObjectLessFitnessOld);
	for (i=0; i<2*l; i++) v[i].rankOld = i+1;

	// compute rank using reevaluated fitness
	std::stable_sort(v.begin(), v.end(), &rankingObjectLessFitnessNew);
	for (i=0; i<2*l; i++) v[i].rankNew = i+1;

	// compute absoulte rank change, i.e., how many ranks lie strictly
	// between old and new rank (step 4)
	for (i=0; i<2*l; i++) {
		v[i].Delta = (unsigned) abs(int(v[i].rankNew) - int(v[i].rankOld));
		if (v[i].Delta) v[i].Delta -= 1;
	}

	// compute uncertainty level (step 5)
	double s = 0.;
	for (i=0; i<2*l; i++) {
		if(v[i].reev) { // loop over reevaluated individuals
			// avaerage the fitness of the reevaluated individuals
			// this is done instead of the re-ranking (step 6)
			p[ v[i].index] .setFitness(.5*(v[i].fitnessNew + v[i].fitnessOld));
			s += 2 * v[i].Delta;
			if(v[i].fitnessNew > v[i].fitnessOld) {
				s -= deltaLimTheta(v[i].rankNew - 1, l, theta);
				s -= deltaLimTheta(v[i].rankOld, l, theta);
			} else if(v[i].fitnessOld > v[i].fitnessNew) {
				s -= deltaLimTheta(v[i].rankNew, l, theta);
				s -= deltaLimTheta(v[i].rankOld - 1, l, theta);
			} else{
				s -= deltaLimTheta(v[i].rankNew, l, theta);
				s -= deltaLimTheta(v[i].rankOld, l, theta);
			}
		}
	}
	s /= l_reev;

	return s;
}
