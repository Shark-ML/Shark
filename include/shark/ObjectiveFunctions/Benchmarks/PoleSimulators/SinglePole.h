/*!
 *
 * \brief Pole balancing simulation for double pole
 *
 * Class for simulating a single pole balancing on a cart.
 * Based on code written by Verena Heidrich-Meisner for the paper
 *
 * V. Heidrich-Meisner and C. Igel. Neuroevolution strategies for
 * episodic reinforcement learning. Journal of Algorithms,
 * 64(4):152–168, 2009.
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_SUTTON_ANDERSON
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_POLE_SUTTON_ANDERSON

#include <math.h>
#include <shark/LinAlg/Base.h>


namespace shark {

class SinglePole {
public:

	//! Convert degrees to radians
	static double degrad(double x) {
		return x * M_PI / 180;
	};

	//! Number of variables
	unsigned noVars() const {
		if(m_isMarkov) 
			return 4;
		return 2;
	}

	//! \param markovian Whether to return velocities in getState
	//! \param normalize Whether to normalize return values in getState
	SinglePole(bool markovian, bool normalize = true) {
		this->m_isMarkov = markovian;

		// current normalization constants are the same as used for
		// the double pole simulation in Heidrich-Meisner code
		// perhaps tweak them for use with single pole?
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

	//! \brief Initialize with specific pole angle
	//! \param state2init initial pole angle (in degrees)
	void initDegree(double state2init)  {
		init(state2init * M_PI / 180);
	}


	//! \brief Initialize with specific pole angle
	//! \param state2init initial pole angle (in radians)
	void init(double state2init = 0.07) {
		m_state[0] = m_state[1] = m_state[3] = 0.0;
		m_state[2] = state2init;
	}

	//! \brief Initialize full m_state
	//! \param a initial cart position
	//! \param b initial cart velocity
	//! \param c initial pole angle (in radians)
	//! \param d initial pole angular velocity
	void init(double a, double b, double c, double d)  {
		m_state[0] = a;
		m_state[1] = b;
		m_state[2] = c;
		m_state[3] = d;
	}

	//! \brief Place m_state in a vector
	//! \param v vector to place m_state in (assumed to be correct size already)
	void getState(RealVector &v) {
		// normalize the m_state variables
		if(m_isMarkov) {
			v(0) = m_state[0] / m_normal_cart;
			v(1) = m_state[1] / m_normal_velo;
			v(2) = m_state[2] / m_normal_pole;
			v(3) = m_state[3] / m_normal_velo;
		}
		else {
			v(0) = m_state[0] / m_normal_cart;
			v(1) = m_state[2] / m_normal_pole;
		}
	}

	//! Returns true when this pole is in an illegal position
	bool failure() {
		const double twelve_degrees =  degrad(12);
		if (m_state[0] < -2.4 || m_state[0] > 2.4  || m_state[2] < -twelve_degrees ||
		        m_state[2] > twelve_degrees) return true;
		return false;
	}

	//! \brief Move the pole with some force
	//! \param output Force to apply. Expects values in [0,1], where values below 0.5 indicate applying force towards the left side and values above indicate force towards the right.
	void move(double output) {
		double dydx[4];

		dydx[1] = 0.;
		dydx[3] = 0.;

		for(int i = 0; i < 2; i++) {
			dydx[0] = m_state[1];
			dydx[2] = m_state[3];
			step(output,m_state,dydx);
			rk4(output,m_state,dydx,m_state);
		}
	}

private:
	void step(double output, double *st, double *derivs) {

		const double MUP = 0.0;
		const double GRAVITY = -9.8;
		const double MASSCART = 1.0;
		const double MASSPOLE = 0.1;
		const double LENGTH = 0.5;
		const double FORCE_MAG = 10.0;

		double force = (output -0.5) * FORCE_MAG * 2;
		double costheta = cos(st[2]);
		double sintheta = sin(st[2]);
		double gsintheta = GRAVITY * sintheta;

		double ml = LENGTH * MASSPOLE;
		double temp = MUP * st[3] / ml;
		double fi = (ml * st[3] * st[3] * sintheta) + (0.75 * MASSPOLE * costheta * (temp + gsintheta));
		double mi = MASSPOLE * (1 - (0.75 * costheta * costheta));

		derivs[1] = (force + fi) / (mi + MASSCART);

		derivs[3] = -0.75 * (derivs[1] * costheta + gsintheta + temp) / LENGTH;
	}

	void rk4(double f,double y[], double dydx[], double yout[]) {
		const double TAU = 0.01;

		double dym[4],dyt[4],yt[4];

		double hh = TAU*0.5;
		double h6 = TAU/6.0;
		for (int i = 0; i <= 3; i++) {
			yt[i] = y[i]+hh*dydx[i];
		}
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		for (int i = 0; i <= 3; i++) {
			yt[i] = y[i]+hh*dyt[i];
		}
		step(f,yt,dym);
		dym[0] = yt[1];
		dym[2] = yt[3];
		for (int i = 0; i <= 3; i++) {
			yt[i] = y[i] + TAU * dym[i];
			dym[i] += dyt[i];
		}
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		for (int i = 0; i <= 3; i++) {
			yout[i] = y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
		}
	}

	double m_state[4];
	bool m_isMarkov;
	double m_normal_cart;
	double m_normal_pole;
	double m_normal_velo;

};

}
#endif
