
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

function InitTutorial(sLogo, sLink) {
	document.write("<h1>" + document.title + "</h1>\n");
	document.write( "<div style=\"width: 600px; height: 400px; text-align:center;\">" );
	document.write( "<iframe style=\"border: 0px black\" id=\"TutorialContent\" align=\"center\" height=\"400px\" width=\"600px\" type=\"text/html\"></iframe>" );
	/*document.write( "<a id=\"toprev\"><img style=\"float: left\" src=\"../prev.png\"/></a>" );
	  document.write( "<a id=\"tonext\"><img style=\"float: right\" src=\"../next.png\"/></a>" );*/

	document.write( "<a id=\"toprev\"><div id=\"prev\">Prev</div></a>" );
	document.write( "<a id=\"tonext\"><div id=\"next\">Next</div></a>" );
	document.write( "</div>" );
}

function AddPage(sTitle, sFilename)
{
	i = g_Filename.length;
	g_Filename.push(sFilename);

	/*txt = document.createTextNode(sTitle);
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
	table.appendChild(row);*/
}

function GotoPage(number)
{
	// set the colors of the links
	/*table = document.getElementById("navi");
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
		}*/

	// change the bottom buttons
	prv = number - 1; if (prv < 0) prv = 0;
	nxt = number + 1; if (nxt >= g_Filename.length) nxt = g_Filename.length - 1;
// 	document.getElementById("tofirst").href = "javascript: GotoPage(0);";
	document.getElementById("toprev").href = "javascript: GotoPage(" + prv + ");";
	document.getElementById("tonext").href = "javascript: GotoPage(" + nxt + ");";
// 	document.getElementById("tolast").href = "javascript: GotoPage(" + (n - 1) + ");";

	// set the content
	document.getElementById("TutorialContent").src = g_Filename[number];
}
