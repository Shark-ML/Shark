//===========================================================================
/*!
 *  \file PCAtest.cpp
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \author Tobias Glasmachers
 *  \date 2007
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


#include <Rng/GlobalRng.h>
#include <ReClaM/LinearModel.h>
#include <ReClaM/PCA.h>
// #include <ReClaM/Dataset.h>


int main(int argc, char** argv)
{
	int i;
	int x, y;

	// generate Gaussian
	Array<double> data(26, 2);
	for (i=0; i<26; i++)
	{
		double a = Rng::gauss();
		double b = Rng::gauss();
		data(i, 0) = 18.0 * a + b + 39.5;
		data(i, 1) = a + 4.5 * b + 9.5;
	}

	// construct PCA model for dimension reduction
	AffineLinearMap model(2, 1);
	PCA optimizer;
	optimizer.init(model);

	// train the model
	printf("PCA training ..."); fflush(stdout);
	optimizer.optimize(model, data);
	printf(" done.\n");

	// transform the data, that is, reduce the dimension from 2 to 1
	Array<double> reduced(26, 1);
	model.model(data, reduced);

	// ascii art visualization of the dataset and the reduced set
	Array<char> twodim(79, 20);
	Array<char> onedim(79);
	onedim = 32;
	twodim = 32;
	for (i=0; i<26; i++)
	{
		x = (int)data(i, 0);
		y = (int)data(i, 1);
		if (x >= 0 && x < 79 && y >= 0 && y < 40)
		{
			if (twodim(x, y) == 32) twodim(x, y) = 'a' + i;
			else twodim(x, y) = '#';
		}
		x = (int)(39.0 + reduced(i, 0));
		if (x >= 0 && x < 79)
		{
			if (onedim(x) == 32) onedim(x) = 'a' + i;
			else onedim(x) = '#';
		}
	}
	printf("two dimensional input:\n");
	printf("-------------------------------------------------------------------------------\n");
	for (y=0; y<20; y++)
	{
		for (x=0; x<79; x++) printf("%c", twodim(x, y));
		printf("\n");
	}
	printf("-------------------------------------------------------------------------------\n");
	printf("one dimensional output:\n");
	printf("-------------------------------------------------------------------------------\n");
	for (x=0; x<79; x++)
	{
		printf("%c", onedim(x));
	}
	printf("\n");
	printf("-------------------------------------------------------------------------------\n");
	printf("\n\n");

	// lines below are for self-testing this example, please ignore
	if (fabs(reduced(0, 0) - 31.87569165) <= 1e-6) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
