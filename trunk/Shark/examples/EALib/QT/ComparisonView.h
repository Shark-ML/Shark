//===========================================================================
/*!
 *  \file ComparisonView.h
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


#ifndef _ComparisonView_H_
#define _ComparisonView_H_


#include <QPixmap>
#include "View.h"
#include "Experiment.h"


class PropertyDesc;


// view for plotting one or several real values over time
class ComparisonView : public View
{
	Q_OBJECT

public:
	ComparisonView(PropertyDesc* property, int mode, QWidget* parent = NULL);
	~ComparisonView();

	void AddExperiment(Experiment* experiment);

	void OnReset();
	void OnChanged(int evals);

	QSize sizeHint() const;

public slots:
// 	void SaveAsScriptFile(bool);
	void SaveAsDataFile(bool);

protected:
	struct tCurve
	{
		Experiment* experiment;
		unsigned int color;
		int mintime;
		int propindex;
	};
	std::vector<tCurve> m_curve;

	static char m_modename[5][50];
	static unsigned int m_default[10];

	PropertyDesc* m_property;
	int m_mode;

	bool m_bNeedsUpdate;

	int maxtime;
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
	void CollectData(Array<double>& data);
	void ComputePlot();

	int TransformY(double value);
	void paintLegend(QPainter& painter);
	void paintCoordinates(QPainter& painter);
	void paintCurve(QPainter& painter, tCurve* curve, const Array<double>& values);

	void paintEvent(QPaintEvent* event);
	void resizeEvent(QResizeEvent* event);
	void mousePressEvent(QMouseEvent* event);

	QPixmap* bitmap;
};


#endif
