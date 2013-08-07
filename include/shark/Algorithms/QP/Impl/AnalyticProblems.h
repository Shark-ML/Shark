/*!
 *  \brief Analytic solutions for special problems
 *
 *  \author  T. Glasmachers, O.Krause
 *  \date    2013
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ALGORITHMS_QP_IMPL_ANALYTICPROBLEMS_H
#define SHARK_ALGORITHMS_QP_IMPL_ANALYTICPROBLEMS_H

namespace shark{ namespace detail{
//! Exact solver for the one-dimensional sub-problem<br>
//! maximize \f$ g \alpha - Q/2 \alpha^2 \f$<br>
//! such that \f$ 0 \leq \alpha \leq U \f$<br>
void solveQuadraticEdge(double& alpha, double g, double Q, double L, double U)
{
	if (Q < 1.e-12)
	{
		if (g > 0.0)
		{
			alpha = U;
		}
		else
		{
			alpha = L;
		}
		return;
	}
	
	alpha = alpha + g / Q;
	alpha = std::min(std::max(alpha,L),U);
}

/// Exact solver for a two-dimensional quadratic sub-problem. with box constraints.
/// The method finds the optimal alpha
void solveQuadratic2DBox(
	double& alphai, double& alphaj,
	double gi, double gj,
	double Qii, double Qij, double Qjj,
	double Li, double Ui, 
	double Lj, double Uj
){
	// try the free solution first if the matrix has full rank
	double detQ = Qii * Qjj - Qij * Qij;
	if(detQ > 1.e-12){
		double mui = (Qjj * gi - Qij * gj) / detQ;
		double muj = (Qii * gj - Qij * gi) / detQ;
		double opti = alphai + mui;
		double optj = alphaj + muj;
		if (opti > Li && optj > Lj && opti < Ui && optj < Uj){
			alphai = opti;
			alphaj = optj;
			return;
		}
	}

	// compute the solution of all four edges
	struct EdgeSolution
	{
		double alphai;
		double alphaj;
	};
	EdgeSolution solution[4];
	
	// edge \alpha_1 = 0
	solution[0].alphai = Li;
	solution[0].alphaj = alphaj;
	solveQuadraticEdge(solution[0].alphaj, gj - Qij * (Li-alphai), Qjj, Lj, Uj);
	
	// edge \alpha_2 = 0
	solution[1].alphai = alphai;
	solution[1].alphaj = Lj;
	solveQuadraticEdge(solution[1].alphai, gi - Qij * (Lj-alphaj), Qii, Li, Ui);
	
	// edge \alpha_1 = U_1
	solution[2].alphai = Ui;
	solution[2].alphaj = alphaj;
	solveQuadraticEdge(solution[2].alphaj, gj - Qij * (Ui-alphai), Qjj, Lj, Uj);
	
	// edge \alpha_2 = U_2
	solution[3].alphai = alphai;
	solution[3].alphaj = Uj;
	solveQuadraticEdge(solution[3].alphai, gi - Qij * (Uj-alphaj), Qii, Li, Ui);
	
	//find the best edge solution
	double maxGain = 0;
	std::size_t maxIndex = 0;
	for(std::size_t k = 0; k != 4; ++k){
		double mui = solution[k].alphai - alphai;
		double muj = solution[k].alphaj - alphaj;
		double gain = mui * (gi - 0.5 * (Qii*mui + Qij*muj))+ muj * (gj - 0.5 * (Qij*mui + Qjj*muj));
		if(gain > maxGain){
			maxIndex = k;
			maxGain = gain;
		}
	}
	
	alphai = solution[maxIndex].alphai;
	alphaj = solution[maxIndex].alphaj;

}
}}

#endif