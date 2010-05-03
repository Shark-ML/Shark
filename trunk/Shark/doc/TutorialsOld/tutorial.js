
//
// javascript methods for the Shark tutorials
//


var g_Filename = new Array();

function Element(parent, sName)
{
	ret = document.createElement(sName);
	parent.appendChild(ret);
	return ret;
}

function InitTutorial(sLogo, sLink)
{
	if (sLink)
	{
		document.write("<table id=\"topleft\"><tr><td><a target=\"_blank\" href=\"" + sLink + "\"><img src=\"" + sLogo + "\"></a></td></tr></table>\n");
	}
	else
	{
		document.write("<div id=\"topleft\"><img src=\"" + sLogo + "\"></div>\n");
	}
	document.write("<div id=\"left\"><table id=\"navi\"></table></div>");
	document.write("<div id=\"main\">\n");
	document.write("<h1>" + document.title + "</h1>\n");
	document.write("<iframe id=\"content\" name=\"contentframe\"><h1>Sorry - your browser does not support embedded frames (iframes).</h1></iframe>\n");
	document.write("<table id=\"bottom\">\n");
	document.write("<tr>\n");
// 	document.write("<td width=\"50\"><a id=\"tofirst\"><img class=\"naviimage\" src=\"../first.png\"></a></td>\n");
	document.write("<td width=\"50\"><a id=\"toprev\"><img class=\"naviimage\" src=\"../prev.png\"></a></td>\n");
	document.write("<td></td>\n");
	document.write("<td width=\"50\"><a id=\"tonext\"><img class=\"naviimage\" src=\"../next.png\"></a></td>\n");
// 	document.write("<td width=\"50\"><a id=\"tolast\"><img class=\"naviimage\" src=\"../last.png\"></a></td>\n");
	document.write("</tr>\n");
	document.write("</table>\n");
	document.write("<div class=\"tabs\" id=\"sharktabs\">\n");
	document.write("<ul>\n");
	document.write("<li><a href=\"../../index.html\"><span>Shark&nbsp;Main&nbsp;Page</span></a></li>\n");
	document.write("<li><a href=\"../../Array/index.html\"><span>Array</span></a></li>\n");
	document.write("<li><a href=\"../../Rng/index.html\"><span>Rng</span></a></li>\n");
	document.write("<li><a href=\"../../LinAlg/index.html\"><span>LinAlg</span></a></li>\n");
	document.write("<li><a href=\"../../FileUtil/index.html\"><span>FileUtil</span></a></li>\n");
	document.write("<li><a href=\"../../EALib/index.html\"><span>EALib</span></a></li>\n");
	document.write("<li><a href=\"../../MOO-EALib/index.html\"><span>MOO-EALib</span></a></li>\n");
	document.write("<li><a href=\"../../ReClaM/index.html\"><span>ReClaM</span></a></li>\n");
	document.write("<li><a href=\"../../Mixture/index.html\"><span>Mixture</span></a></li>\n");
	document.write("<li><a href=\"../../tutorials/index.html\"><span>Tutorials</span></a></li>\n");
	document.write("<li><a href=\"../../faq/index.html\"><span>FAQ</span></a></li>\n");
	document.write("</ul>\n");
	document.write("</div>\n");
	document.write("</div>\n");
}

function AddPage(sTitle, sFilename)
{
	i = g_Filename.length;
	g_Filename.push(sFilename);

	txt = document.createTextNode(sTitle);
	lnk = document.createElement("a");
	cell = document.createElement("td");
	row = document.createElement("tr");
	table = document.getElementById("navi");
	cell.style.backgroundColor = table.style.backgroundColor;

	lnk.href = "javascript: GotoPage(" + i + ");";
	lnk.style.textDecoration = "none";
	lnk.appendChild(txt);
	cell.appendChild(lnk);
	row.appendChild(cell);
	table.appendChild(row);
}

function GotoPage(number)
{
	// set the colors of the links
	table = document.getElementById("navi");
	n = 0;
	for (i=0; i<table.childNodes.length; i++)
	{
		row = table.childNodes[i];
		if (row.nodeType != 1) continue;
		if (row.nodeName != "TR") continue;
		cell = row.childNodes[0];
		if (n == number)
		{
			cell.style.backgroundColor = "#e09060";
		}
		else
		{
			cell.style.backgroundColor = table.style.backgroundColor;
		}
		n++;
	}

	// change the bottom buttons
	prv = number - 1; if (prv < 0) prv = 0;
	nxt = number + 1; if (nxt >= n) nxt = n - 1;
// 	document.getElementById("tofirst").href = "javascript: GotoPage(0);";
	document.getElementById("toprev").href = "javascript: GotoPage(" + prv + ");";
	document.getElementById("tonext").href = "javascript: GotoPage(" + nxt + ");";
// 	document.getElementById("tolast").href = "javascript: GotoPage(" + (n - 1) + ");";

	// set the content
	document.getElementById("content").src = g_Filename[number];
}
