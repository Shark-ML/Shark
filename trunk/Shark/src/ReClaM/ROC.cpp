//===========================================================================
/*!
 *  \file ROC.cpp
 *
 *  \brief computes a "receiver operator characteristics" curve
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
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


#include <algorithm>
#include <ReClaM/ROC.h>


ROC::ROC(Model& model, const Array<double>& input, const Array<double>& target)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);

	int t, T = input.dim(0);
	SIZE_CHECK(target.dim(0) == (unsigned)T);
	SIZE_CHECK(target.dim(1) == 1);

	int tp, tn;
	int positive = 0;
	int negative = 0;
		for (t = 0; t < T; t++) if (target(t, 0) > 0.0) positive++; else negative++;
	score_positive.resize(positive);
	score_negative.resize(negative);

	// compute scores
	Array<double> output;
	double value;

	tp = 0; tn = 0;
	for (t = 0; t < T; t++)
	{
		model.model(input[t], output);
		value = output(0);
		if (target(t, 0) > 0.0)
		{
			score_positive[tp] = value;
			tp++;
		}
		else
		{
			score_negative[tn] = value;
			tn++;
		}
	}
	if (tp != positive || tn != negative) throw SHARKEXCEPTION("[ROC::ROC] internal error");

	// sort positives and negatives by score
	std::sort(score_positive.begin(), score_positive.end());
	std::sort(score_negative.begin(), score_negative.end());
}

double ROC::Threshold(double falseAcceptanceRate)
{
	double ii = (1.0 - falseAcceptanceRate) * score_negative.size();
	int i = (unsigned int)ii;
	if (i >= (int)score_negative.size()) return 1e100;
	else if (i < 0) return -1e100;
	else if (i == ii || i == (int)score_negative.size() - 1) return score_negative[i];

	// linear interpolation
	double rest = ii - i;
	return (1.0 - rest) * score_negative[i] + rest * score_negative[i + 1];
}

double ROC::Value(double falseAcceptanceRate)
{
	double threshold = Threshold(falseAcceptanceRate);
	unsigned int i;

	// "verification rate" = 1.0 - "false rejection rate"
	// TODO: build binary search!
	for (i = 0; i < score_positive.size() && score_positive[i] < threshold; i++);
	if (i == 0) return 1.0;
	else if (i == score_positive.size()) return 0.0;

	// linear interpolation
	double sl = score_positive[i - 1];
	double sr = score_positive[i];
	double inter = ((threshold - sl) * i + (sr - threshold) * (i - 1)) / (sr - sl);
	return 1.0 - inter / score_positive.size();
}

double ROC::EqualErrorRate()
{
	double threshold;
	int i, c = 0;

	double e1 = 0.0;
	double e2 = 0.0;

	double dc = score_positive.size();
	double di = score_negative.size();

	for (i = 0; i < (int)score_negative.size(); i++)
	{
		threshold = score_negative[i];
		for ( ;c < (int)score_positive.size() && score_positive[c] < threshold; c++);

		e1 = i / di;			// type 1 error
		e2 = 1.0 - c / dc;		// type 2 error

		if (e1 >= e2) break;
	}
	return 0.5 *(e1 + e2);
}

