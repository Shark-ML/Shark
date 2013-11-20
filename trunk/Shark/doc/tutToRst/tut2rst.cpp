//===========================================================================
/*!
 *  \brief Small tool for preparing the tutorials.
 *
 *  \author  T. Glasmachers
 *  \date    2013
 *
 *
 *  \par
 *  This program requires two command line parameters:
 *     filename
 *     sharkpath
 *  The first of these, after appending the extension ".tut", is
 *  the full path to a tutorial template file. The program reads
 *  this file and parses it for macros of the form
 *     "..sharkcode<file,name>"
 *  Such a macro is replaced with a corresponding code snippet from
 *  the specified file (a path relative to sharkpath). The file is
 *  assumed to contain special marker comments as follows:
 *     //###begin<name>
 *     ... code to be inserted ...
 *     //###end<name>
 *  Here, "name" is any token, possibly with an index, for telling
 *  multiple code snippets within a single file apart. There may be
 *  multiple snippets of the same name. In this case the snippets
 *  are concatenated and inserted together.
 *  After all replacements have been made the new content is saved
 *  to a file with extension ".rst", which is ready for processing
 *  with sphinx.
 *  The shorthand form
 *     "..sharkcode<file>"
 *  is recognized for including the whole source file.
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */
//===========================================================================

#include <string>
#include <fstream>
#include <iostream>


using namespace std;


string readFile(string filename)
{
	ifstream ifs(filename.c_str(), ios_base::binary);
	string content = string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	if (! ifs.good()) throw string("reading file '" + filename + "' failed");
	return content;
}

void help()
{
	cerr << "usage: ./tut2rst <filename> <sharkpath>" << endl;
}

int main(int argc, char** argv)
{
	try
	{
		// parse command line
		if (argc != 3){ help(); return 1;}
		string filename = argv[1];
		string sharkpath = argv[2];
		if (! sharkpath.empty() && sharkpath[sharkpath.size() - 1] != '/') sharkpath += "/";

		cout << "tut2rst: processing " << filename << endl;

		// read input
		string input = readFile(filename + ".tut");
		string output;

		// process
		size_t start = 0;
		while (true)
		{
			size_t pos = input.find("..sharkcode<", start);
			if (pos == string::npos)
			{
				// not found; append remaining input and stop
				output += input.substr(start);
				break;
			}

			// add content until here
			output += input.substr(start, pos - start);

			// parse example filename and snippet name
			pos += 12;
			size_t comma = input.find(",", pos);
			size_t closing = input.find(">", pos);
			start = closing + 1;
			if (closing == string::npos) throw string("sharkcode tag not closed (missing '>')");
			if (comma > closing)
			{
				// simple form for the inclusion of whole files
				string examplename = input.substr(pos, closing - pos);
				cout << "  placing file '" << examplename << "'" << endl;

				// insert file contents; increase indentation level
				string example = readFile(sharkpath + examplename);
				for (std::size_t i=0; i<example.size(); i++)
				{
					if (example[i] == '\n') output += "\n\t";
					else output += example[i];
				}
			}
			else
			{
				// complex form with name tag
				string examplename = input.substr(pos, comma - pos);
				size_t npos = comma + 1;
				while (input[npos] == ' ') npos++;
				string name = input.substr(npos, closing - npos);
				cout << "  placing snippet '" << name << "' from file '" << examplename << "'" << endl;

				// insert snippet(s) from example file
				string example = readFile(sharkpath + examplename);
				size_t start = 0;
				size_t num = 0;
				while (true)
				{
					// identify snippet
					size_t begin = example.find("//###begin<" + name + ">", start);
					if (begin == string::npos) break;
					size_t end = example.find("//###end<" + name + ">", begin);
					if (end == string::npos) throw string("end marker for '" + name + "' not found in file '" + examplename + "'");
					while (example[begin] != '\n') begin++;
					string snippet = example.substr(begin, end - begin);

					// remove inner snippet markers
					size_t s_start = 0;
					while (true)
					{
						size_t begin1 = snippet.find("//###begin<", s_start);
						size_t begin2 = snippet.find("//###end<", s_start);
						if (begin1 == string::npos && begin2 == string::npos) break;
						size_t begin = min(begin1, begin2);
						size_t end = snippet.find(">", begin);
						if (end == string::npos) throw string("sharkcode tag not closed (missing '>')");
						end++;
						if (snippet[end] == '\n') end++;
						snippet.erase(begin, end - begin);
						s_start = begin;
					}

					// add snippet to output
					for (size_t i=0; i<snippet.size(); i++)
					{
						char c = snippet[i];
						if (c == '\n') output += "\n\t";    // make sure code is indented
						else output += c;
					}
					start = end;
					num++;
				}
				if (num == 0) throw string("begin marker for '" + name + "' not found in file '" + examplename + "'");
			}
		}

		// write output
		ofstream ofs((filename + ".rst").c_str(), ios_base::binary);
		ofs.write(output.c_str(), output.size());
		cout << "done." << endl;
	}
	catch (string const& ex)
	{
		cout << "  ERROR: " << ex << endl;
		return 1;
	}
}
