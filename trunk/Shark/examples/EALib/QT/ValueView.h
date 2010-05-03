//===========================================================================
/*!
 *  \file ValueView.h
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


#ifndef _ValueView_H_
#define _ValueView_H_


#include <QPixmap>
#include "View.h"


class PropertyDesc;


// view for plotting one or several real values over time
class ValueView : public View
{
	Q_OBJECT

public:
	ValueView(QString title, Experiment* experiment, PropertyDesc* property, QWidget* parent = NULL);
	~ValueView();

	void OnReset();
	void OnChanged(int evals);

	QSize sizeHint() const;

protected:
	PropertyDesc* m_property;
	int m_totalEvaluations;

	Array<int> m_evals;
	Array<double> m_values;

	QColor m_color;
	int m_style;

	double minval;
	double maxval;

	int effectiveWidth;
	int effectiveHeight;

	int unitX;					// iterations per horizontal unit
	int unitwidth;				// pixels per horizontal unit
	int numunitsx;				// number of horizontally drawn units
	double minrange;			// minimal (transformed) display value
	double maxrange;			// maximal (transformed) display value
	double unitY;				// (transformed) vertical unit
	int unitheight;				// pixels per vertical unit
	int numunitsy;				// number of vertically drawn units
	void ComputeRange();

	int TransformX(int iter);
	int TransformY(double value);
	void paintLegend(QPainter& painter);
	void paintCoordinates(QPainter& painter);
	void paintData(QPainter& painter);

	void paintEvent(QPaintEvent* event);
	void resizeEvent(QResizeEvent* event);

	bool performMinimalUpdate;
	int numberOfDrawnPatterns;
	QPixmap* bitmap;
};


#endif
