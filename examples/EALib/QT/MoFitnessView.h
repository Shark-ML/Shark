//===========================================================================
/*!
 *  \file MoFitnessView.h
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


#ifndef _MoFitnessView_H_
#define _MoFitnessView_H_


#include <QPixmap>
#include "View.h"
#include "Optimization.h"


// Visualization of the pareto front (population)
// for problems with two objectives.
class MoFitnessView : public View
{
	Q_OBJECT

public:
	MoFitnessView(QString title, Experiment* experiment, QWidget* parent = NULL);
	~MoFitnessView();

	inline void setMinMax(double min1, double max1, double min2, double max2)
	{
		minval[0] = min1;
		maxval[0] = max1;
		minval[1] = min2;
		maxval[1] = max2;
	}

	void OnReset();
	void OnChanged(int evals);

	QSize sizeHint() const;

protected:
	void paintEvent(QPaintEvent* event);
	void resizeEvent(QResizeEvent* event);

	void ComputeRange(double m, double M, int space, int mindist, double& start, double& step, int& N);
	void paintCoordinates(QPainter& painter);
	void paintData(QPainter& painter);

	double minval[2];
	double maxval[2];

	bool performMinimalUpdate;			// speed up painting
	unsigned int m_drawn;				// number of points already drawn
	Array<double> m_point;				// point coordinates

	QPixmap* bitmap;
};


#endif
