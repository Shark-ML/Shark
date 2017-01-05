/*!
 *
 * \brief Pole balancing simulation for double poles
 *
 * Class for simulating two poles balancing on a cart.
 * Based on code written by Verena Heidrich-Meisner for the paper
 *
 * V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for episodic reinforcement learning. Journal of Algorithms, 64(4):152–168, 2009.
 *
 * which was in turn based on code available at http://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
 * as of 2015/4/19, written by Rich Sutton and Chuck Anderson and later modified.
 * Faustino Gomez wrote the physics code using the differential equations from
 * Alexis Weiland's paper and added the Runge-Kutta solver.
 *
 * \author      Johan Valentin Damgaard
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 *
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_TEXAS
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_TEXAS

#include <math.h>
#include <shark/LinAlg/Base.h>



namespace shark {

class DoublePole {
public:

	//! Number of variables
	unsigned noVars() const {
		if(m_isMarkovian)
			return 6;
		return 3;
	}

	//! \param markovian Whether to return velocities in getState
	//! \param normalize Whether to normalize return values in getState
	DoublePole(bool markovian, bool normalize = true) {
		m_isMarkovian = markovian;

		// current normalization constants are the same as used
		// in the Heidrich-Meisner code
		// perhaps tweak them?
		if(normalize) {
			this->m_normal_cart = 4.8;
			this->m_normal_pole = 0.52;
			this->m_normal_velo = 2;
		}
		else {
			this->m_normal_cart = 1.;
			this->m_normal_pole = 1.;
			this->m_normal_velo = 1.;
		}

	}


	//! \brief Place m_state in a vector
	//! \param v vector to place m_state in (assumed to be correct size already)
	void getState(RealVector &v) {
		// normalize the m_state variables.
		if(m_isMarkovian) {
			v(0) = m_state[0] / m_normal_cart;
			v(1) = m_state[1] / m_normal_velo;
			v(2) = m_state[2] / m_normal_pole;
			v(3) = m_state[3] / m_normal_velo;
			v(4) = m_state[4] / m_normal_pole;
			v(5) = m_state[5] / m_normal_velo;
		}
		else {
			v(0) = m_state[0] / m_normal_cart;
			v(1) = m_state[2] / m_normal_pole;
			v(2) = m_state[4] / m_normal_pole;
		}
	}

	//! \brief Initialize with specific angle for large pole
	//! \param state2init initial pole angle (in radians)
	void init(double state2init = 0.07)  {
		m_state[0] = m_state[1] = m_state[3] = m_state[4] = m_state[5] = 0;
		m_state[2] = state2init;
	}

	//! \brief Initialize with specific angle for large pole
	//! \param state2init initial pole angle (in degrees)
	void initDegree(double state2init)  {
		init(state2init * M_PI / 180);
	}

	//! \brief Initialize full m_state
	//! \param a initial cart position
	//! \param b initial cart velocity
	//! \param c initial large pole angle (in radians)
	//! \param d initial large pole angular velocity
	//! \param e initial small pole angle (in radians)
	//! \param f initial small pole angular velocity
	void init(double a, double b, double c, double d, double e = 0., double f = 0.)  {
		m_state[0] = a;
		m_state[1] = b;
		m_state[2] = c;
		m_state[3] = d;
		m_state[4] = e;
		m_state[5] = f;
	}

	//! Returns true when this pole is in an illegal position
	bool failure()
	{
		const double thirty_six_degrees= 0.628329;
		const double failureAngle = thirty_six_degrees;

		return(m_state[0] < -2.4              ||
		       m_state[0] > 2.4               ||
		       m_state[2] < -failureAngle     ||
		       m_state[2] > failureAngle      ||
		       m_state[4] < -failureAngle     ||
		       m_state[4] > failureAngle);
	}

	//! Return "jiggle", abstract representation of how much the the cart oscillates
	double getJiggle() {
		return (std::abs(m_state[0]) + std::abs(m_state[1]) + std::abs(m_state[2]) + std::abs(m_state[3]));
	}

	//! \brief Move the pole with some force
	//! \param output Force to apply. Expects values in [0,1], where values below 0.5 indicate applying force towards the left side and values above indicate force towards the right.
	void move(double output)
	{
		double  dydx[6];
		const double TAU= 0.01;

		const bool useRK4=true;
		if(useRK4) {
			for(int i=0; i<2; ++i) {
				dydx[0] = m_state[1];
				dydx[2] = m_state[3];
				dydx[4] = m_state[5];
				step(output,m_state,dydx);
				rk4(output,m_state,dydx,m_state);
			}
		}
		else {
			const double EULER_STEPS = 4.;
			const double EULER_TAU= TAU/EULER_STEPS;
			for(int i=0; i<2 * EULER_STEPS; ++i) {
				step(output,m_state,dydx);
				m_state[0] += EULER_TAU * m_state[1];
				m_state[1] += EULER_TAU * dydx[1];
				m_state[2] += EULER_TAU * m_state[3];
				m_state[3] += EULER_TAU * dydx[3];
				m_state[4] += EULER_TAU * m_state[5];
				m_state[5] += EULER_TAU * dydx[5];
			}
		}
	}

private:
	void step(double action, double *st, double *derivs)
	{
		const double MUP = 0.000002;
		const double MUC = 0.0005;
		const double GRAVITY= -9.8;
		const double MASSCART= 1.0;
		const double LENGTH_1   = 0.5;
		const double MASSPOLE_1 = 0.1;
		const double LENGTH_2   = 0.1 * LENGTH_1;
		const double MASSPOLE_2 = 0.1 * MASSPOLE_1;
		const double FORCE_MAG= 10.0;

		double signum = 0.;

		if (st[1] < 0) {
			signum = -1;
		}
		if (st[1] > 0) {
			signum = 1;
		}

		double force =  ((action - 0.5) * FORCE_MAG * 2) - (MUC*signum);
		double costheta_1 = cos(st[2]);
		double sintheta_1 = sin(st[2]);
		double gsintheta_1 = GRAVITY * sintheta_1;
		double costheta_2 = cos(st[4]);
		double sintheta_2 = sin(st[4]);
		double gsintheta_2 = GRAVITY * sintheta_2;

		double ml_1 = LENGTH_1 * MASSPOLE_1;
		double ml_2 = LENGTH_2 * MASSPOLE_2;
		double temp_1 = MUP * st[3] / ml_1;
		double temp_2 = MUP * st[5] / ml_2;
		double fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
		       (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
		double fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
		       (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
		double mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
		double mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));

		derivs[1] = (force + fi_1 + fi_2)
		            / (mi_1 + mi_2 + MASSCART);

		derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
		            / LENGTH_1;
		derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
		            / LENGTH_2;

	}

	void rk4(double f, double y[], double dydx[], double yout[])
	{
		const double TAU= 0.01;
		double dym[6],dyt[6],yt[6];

		double hh=TAU*0.5;
		double h6=TAU/6.0;
		for (int i=0; i<=5; i++) yt[i]=y[i]+hh*dydx[i];
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		dyt[4] = yt[5];
		for (int i=0; i<=5; i++) yt[i]=y[i]+hh*dyt[i];
		step(f,yt,dym);
		dym[0] = yt[1];
		dym[2] = yt[3];
		dym[4] = yt[5];
		for (int i=0; i<=5; i++) {
			yt[i]=y[i]+TAU*dym[i];
			dym[i] += dyt[i];
		}
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		dyt[4] = yt[5];
		for (int i=0; i<=5; i++)
			yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
	}

	bool m_isMarkovian;

	double m_state[6];

	double m_normal_cart;
	double m_normal_pole;
	double m_normal_velo;

};
}
#endif
