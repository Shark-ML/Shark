/*!
 * 
 * \file        AnalyticProblems.h
 *
 * \brief       Analytic solutions for special problems
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ALGORITHMS_QP_IMPL_ANALYTICPROBLEMS_H
#define SHARK_ALGORITHMS_QP_IMPL_ANALYTICPROBLEMS_H

namespace shark{ namespace detail{
//! Exact solver for the one-dimensional sub-problem<br>
//! maximize \f$ g \alpha - Q/2 \alpha^2 \f$<br>
//! such that \f$ 0 \leq \alpha \leq U \f$<br>
inline void solveQuadraticEdge(double& alpha, double g, double Q, double L, double U)
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
	
	alpha += g / Q;
	alpha = std::min(std::max(alpha,L),U);
}

/// Exact solver for a two-dimensional quadratic sub-problem with box constraints.
/// The method finds the optimal alpha
inline void solveQuadratic2DBox(
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

/// Exact solver for a two-dimensional quadratic sub-problem with simplex constraints.
/// The exact constraints are alphai >= 0, alphaj=>0, alphai+alphaj<maxSum
inline void solveQuadratic2DTriangle(
	double& alphai, double& alphaj,
	double gi, double gj,
	double Qii, double Qij, double Qjj,
	double maxSum
)
{
	// try the free solution first if the matrix has full rank
	double detQ = Qii * Qjj - Qij * Qij;
	if(detQ > 1.e-12){
		double mui = (Qjj * gi - Qij * gj) / detQ;
		double muj = (Qii * gj - Qij * gi) / detQ;
		double opti = alphai + mui;
		double optj = alphaj + muj;
		if (
			opti > 0 && optj > 0 //check that we are in the box
			&& opti + optj < maxSum//and below the diagonal
		){
			alphai = opti;
			alphaj = optj;
			return;
		}
	}

	// otherwise process all edges
	struct EdgeSolution
	{
		double alphai;
		double alphaj;
	};
	EdgeSolution solution[3];
	// edge alphai = 0
	solution[0].alphai = 0;
	solution[0].alphaj = alphaj;
	solveQuadraticEdge(solution[0].alphaj, gj + Qij * alphai, Qjj, 0, maxSum);
	// edge alphaj = 0
	solution[1].alphai = alphai;
	solution[1].alphaj = 0;
	solveQuadraticEdge(solution[1].alphai, gi + Qij * alphaj, Qii, 0, maxSum);
	// edge \alpha_1 + \alpha_2 = U
	{
		//start a linsearch from alphai = maxSum and alphaj=0
		solution[2].alphaj = 0.0;
		double ggi = gi - (maxSum - alphai) * Qii + alphaj * Qij;
		double ggj = gj - (maxSum - alphai) * Qij + alphaj * Qjj;
		solveQuadraticEdge(solution[2].alphaj, ggj - ggi, Qii + Qjj - 2.0 * Qij, 0,maxSum);
		solution[2].alphai = maxSum - solution[2].alphaj;
	}

	//find the best edge solution
	double maxGain = -1;
	std::size_t maxIndex = 0;
	for(std::size_t k = 0; k != 3; ++k){
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

	// improve numerical stability:
	if (alphai < 1e-12 * maxSum) 
		alphai = 0.0;
	else if (maxSum - alphai < 1e-12 * maxSum) { 
		alphai = maxSum;
		alphaj = 0.0;
	}
	if (alphaj < 1e-12 * maxSum) 
		alphaj = 0.0;
	else if (maxSum - alphaj < 1e-12 * maxSum) {
		alphai = 0.0;
		alphaj = maxSum;
	}
}


/// \brief Calculates the maximum value of a 2D quadratic Problem.
///
/// This is used heavily for maximum gain working set selection for box constrained
/// problems. If the matrix is not full rank, a minimum determinant is assumed.
/// The matrix is not allowed to be indefinite (but this is not checked).
inline double maximumGainQuadratic2D(double Qii, double Qjj, double  Qij, double gi, double gj, double minDetFrac = 1.e-12){
	double diagQ = Qii * Qjj;
	double detQ = diagQ - Qij * Qij;
	if(detQ <= minDetFrac*diagQ){
		Qii += 1.e-6;
		Qjj += 1.e-6;
		diagQ = Qii * Qjj;
		detQ = diagQ - Qij * Qij;
	}
	//gain = g^T Q^-1 g
	return (gj*gj*Qii - 2.0*gj*gi*Qij + gi*gi*Qjj) / detQ;
}


/// \brief Calculates the maximum value of a 2D quadratic problem with equality constraint xi+xj=0 and x1 > 0
///
/// This is used heavily for maximum gain working set selection for box constrained
/// problems with additional quality constraint. If the matrix is not full rank, a minmum curvature
/// along the line is assumed. The both constraints together lead to a search direction (1,-1).
/// The matrix is not allowed to be indefinite (but this is not checked).
inline double maximumGainQuadratic2DOnLine(double Qii, double Qjj, double  Qij, double gi, double gj, double minCurvature = 1.e-12){
	//check that this is an ascending direction, otherwise gain is 0
	double g = gi - gj;
	if(g <= 0) return 0.0;
	
	double Q = std::max(Qii + Qjj - 2.0 * Qij,minCurvature);
	return g*g/Q;
}
}}

#endif