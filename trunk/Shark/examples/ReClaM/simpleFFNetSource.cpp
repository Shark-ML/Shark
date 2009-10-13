//===========================================================================
/*!
 *  \file simpleFFNetSource.cpp
 *
 *  \author  C. Igel and M. Toussaint
 *  \date    2004
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
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
 */
//===========================================================================




#include <Array/ArrayIo.h>
#include <FileUtil/IOTools.h>
#include <ReClaM/MSEFFNet.h>
#include <ReClaM/EarlyStopping.h>
#include <ReClaM/NetParams.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/MeanSquaredError.h>
#include <ReClaM/createConnectionMatrix.h>
#include <ReClaM/FFNetSource.h>


using namespace std;


class Net : public MSEFFNet // feed-forward network
{
public:
	Net(const string &filename, double c = 0) : MSEFFNet(filename)
	{ }
	Net(const unsigned in = 0, const unsigned out = 0, double c = 0)
			: MSEFFNet(in, out)
	{ }
	Net(const unsigned in, const unsigned out, const Array<int> &cmat, double c = 0)
			: MSEFFNet(in, out, cmat)
	{ }
	Net(const unsigned in, const unsigned out,
		const Array<int> &cmat, const Array<double>& wmat, double c = 0)
			: MSEFFNet(in, out, cmat, wmat)
	{ }
	double gOutput(double a)
	{
		return a;
	}
	double dgOutput(double ga)
	{
		return 1.;
	}
	double g(double a)
	{
		return a / (1 + fabs(a));
	}
	double dg(double ga)
	{
		return (1 - sgn(ga) * ga) *(1 - sgn(ga) * ga);
	}
	double inline sgn(double x)
	{
		return (x > 0.0) ? 1.0 : -1.0;
	}
};

void model(double *in, double *out) {
  double z0 = in[0];
  double z1 = in[1];
  double z2 = z0 * 0.016556087298737 + z1 * 0.01782074404296;
  z2 = z2 / (1 + fabs(z2));
  double z3 = z0 * 0.092442109947455 + z1 * 0.050230641674723 + -0.0054050702217484;
  z3 = z3 / (1 + fabs(z3));
  double z4 = z0 * -0.057728339990863 + z1 * -0.053469142608909 + 0.048458969068542;
  z4 = z4 / (1 + fabs(z4));
  out[0] = z0 * 0.070491154444552 + z1 * -0.0075326520223182 + z2 * 0.094400800053357 + z3 * -0.0012199356114029 + z4 * -0.014226921338189 + 0.09307984509374;
  out[1] = z0 * 0.037099722840633 + z1 * -0.093781292992451 + z2 * -0.023027826640639 + z3 * -0.032499106025279 + z4 * -0.041246796907915 + 0.072655676292645;
};

bool testModel(Array<double> net_o)
{
	double i[2];  double o[2];
	i[0] = .25;
	i[1] = -.5;
	model(i, o);
	std::cout << o[0] << " " << o[1] << std::endl;

	//for self-testing, please ignore
	if (fabs(net_o(0)-o[0])+fabs(net_o(1)-o[1]) <1.e-14) return true;
	else return false;

}


int main(int argc, char **argv)
{
	unsigned in = 2, out = 2, hid = 3;

	Array<int> con;  // connection matrix
	Array<double> w; // weight matrix

	//for self-testing, please ignore
	bool ok;
	//end self-testing block

	createConnectionMatrix(con, in, hid, out, true, true, true);
	con(in, in + hid + out) = 0;
	w.resize(con);

	Net net(in, out, con, w, 0);

	net.initWeights(-0.1, 0.1);
	w = net.getWeights();
	writeArray(w, cout); cout << endl;


	FFNetSource(cout, in, out, con, w, "# / (1 + fabs(#))", "#", 14);

	Array<double> i;
	Array<double> o;
	i.resize(in);
	o.resize(out);
	i(0) = .25;
	i(1) = -.5;
	net.model(i, o);
	cout << o(0) << " " << o(1) << endl;

	ok=testModel(o);

	// lines below are for self-testing this example, please ignore
	if (ok) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

