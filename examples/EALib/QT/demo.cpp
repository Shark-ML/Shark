//===========================================================================
/*!
 *  \file demo.cpp
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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


// #include "ValueView.h"
// #include "LandscapeView2D.h"
// #include "LandscapeView3D.h"
#include "MainWidget.h"
#include "Optimization.h"
#include "Experiment.h"
#include "Wizard.h"

#include <QApplication>
#include <Rng/GlobalRng.h>


int main(int argc, char** argv)
{
	int ret = EXIT_FAILURE;
	try
	{
// 		// BEGIN TEST
// 		int i;
// 		std::vector<double> A(10);
// 		std::vector<double> B(8);
// 		for (i=0; i<10; i++) A[i] = 2 + 2 * i;
// 		for (i=0; i<8; i++) B[i] = 1 + 2 * i;
// 		double twosided, aLeft, bLeft;
// 		UTest(A, B, twosided, aLeft, bLeft);
// 		printf("2-sided: %g   a-left: %g   b-left: %g\n", twosided, aLeft, bLeft);
// 		exit(0);
// 		// END TEST

		QApplication app(argc, argv);

		MainWidget mainwidget;
		mainwidget.show();

		ret = app.exec();
	}
	catch (const SharkException& e)
	{
		std::cout << "SharkException: " << e.what() << std::endl;
	}

	return ret;
}
