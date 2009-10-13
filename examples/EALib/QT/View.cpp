//===========================================================================
/*!
 *  \file View.cpp
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


#include "View.h"
#include "Experiment.h"
#include "Optimization.h"


View::View(QString title, Experiment* experiment, QWidget* parent)
: QWidget(parent)
{
	m_experiment = experiment;

	setWindowTitle(title);

	connect(this, SIGNAL(delayedDelete()), this, SLOT(onDelete()), Qt::QueuedConnection);
}

View::~View()
{
}


void View::closeEvent(QCloseEvent* event)
{
	emit destroyed();
	emit delayedDelete();
}

void View::reset()
{
	OnReset();
	emit done();
}

void View::changed(int evals)
{
	OnChanged(evals);
	emit done();
}

void View::destroy()
{
	close();
}

void View::onDelete()
{
	// There seems no way for this with QT:
	// delete this;
}
