//===========================================================================
/*!
 *  \file LandscapeView3D.h
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


#ifndef _LandscapeView3D_H_
#define _LandscapeView3D_H_


#include "View.h"
#include "Optimization.h"


class LandscapeView3D : public View
{
	Q_OBJECT

public:
	LandscapeView3D(QString title, Experiment* experiment, QWidget* parent = NULL);
	~LandscapeView3D();

	void OnReset();
	void OnChanged(int evals);

	inline void monitorPopulation() { m_monitorPopulation = true; }
	inline void monitorSearchDistribution() { m_monitorSearchDistribution = true; }

	QSize sizeHint() const;

protected:
	double get(double x, double y);
	void ComputeView();

	void paintEvent(QPaintEvent* event);
	void resizeEvent(QResizeEvent* event);

	bool m_monitorPopulation;
	bool m_monitorSearchDistribution;

	Array<double> m_value;

	QImage m_image;

	// camera position and view coordinates
	double viewposX;
	double viewposY;
	double viewposZ;
	double viewdirXY;
	double viewdirZ;
	double targetposX;
	double targetposY;
	double targetposZ;

	// tables
	double dist[2048];
	double right[256];
	unsigned int m_palette[1024];
};


#endif
