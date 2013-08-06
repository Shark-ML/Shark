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
//! maximize \f$ g \alpha - Q/2 \mu^2 \f$<br>
//! such that \f$ 0 \leq \alpha \leq U \f$<br>
//! The method returns the optimal alpha as well as
//! the step mu leading to the update
//! \f$ \alpha \leftarrow \alpha + \mu \f$.
double stepEdge(double alpha, double g, double Q, double L, double U, double& mu)
{
	// compute the optimal unconstrained step
	double muHat = g / Q;

	// check for numerical problems
	if( !boost::math::isnormal( muHat ) )
	{
		if (g > 0.0) mu = 1e100;
		else mu = -1e100;
		return 1e100;
	}

	// compute the optimal constrained step
	double mu_g;
	if (muHat <= L - alpha)
	{
		mu_g = L - alpha;
		mu = -1e100;
	}
	else if (muHat >= U - alpha)
	{
		mu_g = U - alpha;
		mu = 1e100;
	}
	else
	{
		mu_g = muHat;
		mu = muHat;
	}

	// compute (twice) the gain
	double deltaMu = muHat - mu_g;
	return (muHat * muHat - deltaMu * deltaMu) * Q;
}

/// Exact solver for a two-dimensional sub-problem.
/// If the optimal solution is on the edge, then the
/// corresponding mu-value is set to either +1e100
/// or -1e100 as an indication.
void solve2DBox(
		double alphai, double alphaj,
		double gi, double gj,
		double Qii, double Qij, double Qjj,
		double Li, double Ui, double Lj, double Uj,
		double& mui, double& muj
){
	double QD = Qii * Qjj;
	double detQ = QD - Qij * Qij;
	if (detQ < 1e-10 * QD)
	{
		if (Qii == 0.0 && Qjj == 0.0)
		{
			// Q has rank zero (is the zero matrix)
			// just follow the gradient
			if (gi > 0.0) mui = 1e100;
			else if (gi < 0.0) mui = -1e100;
			else mui = 0.0;
			if (gj > 0.0) muj = 1e100;
			else if (gj < 0.0) muj = -1e100;
			else muj = 0.0;
		}
		else
		{
			// Q has rank one
			double gamma = Qii * gj - Qij * gi;
			double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
			double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

			// edge with fixed mu_i
			if (Qij * gamma > 0.0)
			{
				edgei_mui = -1e100;
				edgei_gain = stepEdge(alphaj, gj - Qij * (Li - alphai), Qjj, Lj, Uj, edgei_muj);
			}
			else if (Qij * gamma < 0.0)
			{
				edgei_mui = 1e100;
				edgei_gain = stepEdge(alphaj, gj - Qij * (Ui - alphai), Qjj, Lj, Uj, edgei_muj);
			}

			// edge with fixed mu_j
			if (Qii * gamma < 0.0)
			{
				edgej_muj = -1e100;
				edgej_gain = stepEdge(alphai, gi - Qij * (Lj - alphaj), Qii, Li, Ui, edgej_mui);
			}
			else if (Qii * gamma > 0.0)
			{
				edgej_muj = 1e100;
				edgej_gain = stepEdge(alphai, gi - Qij * (Uj - alphaj), Qii, Li, Ui, edgej_mui);
			}

			// keep the better edge point
			if (edgei_gain > edgej_gain)
			{
				mui = edgei_mui;
				muj = edgei_muj;
			}
			else
			{
				mui = edgej_mui;
				muj = edgej_muj;
			}
		}
	}
	else
	{
		// Q has full rank of two, thus it is invertible
		double muiHat = (Qjj * gi - Qij * gj) / detQ;
		double mujHat = (Qii * gj - Qij * gi) / detQ;
		double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
		double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

		// edge with fixed mu_i
		if (muiHat < Li - alphai)
		{
			edgei_mui = -1e100;
			edgei_gain = stepEdge(alphaj, gj - Qij * (Li - alphai), Qjj, Lj, Uj, edgei_muj);
		}
		else if (muiHat > Ui - alphai)
		{
			edgei_mui = 1e100;
			edgei_gain = stepEdge(alphaj, gj - Qij * (Ui - alphai), Qjj, Lj, Uj, edgei_muj);
		}

		// edge with fixed mu_j
		if (mujHat < Lj - alphaj)
		{
			edgej_muj = -1e100;
			edgej_gain = stepEdge(alphai, gi - Qij * (Lj - alphaj), Qii, Li, Ui, edgej_mui);
		}
		else if (mujHat > Uj - alphaj)
		{
			edgej_muj = 1e100;
			edgej_gain = stepEdge(alphai, gi - Qij * (Uj - alphaj), Qii, Li, Ui, edgej_mui);
		}

		// keep the unconstrained optimum or the better edge point
		if (edgei_gain == 0.0 && edgej_gain == 0.0)
		{
			mui = muiHat;
			muj = mujHat;
		}
		else if (edgei_gain > edgej_gain)
		{
			mui = edgei_mui;
			muj = edgei_muj;
		}
		else
		{
			mui = edgej_mui;
			muj = edgej_muj;
		}
	}
}
}}

#endif