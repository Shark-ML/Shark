\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{manual}[1998/04/06 v1.0 LaTeX document class for manuals]
\newif\if@mainmatter \@mainmattertrue

%
% paper size A4
%
\setlength\paperheight {297mm}
\setlength\paperwidth  {210mm}

%
% two side
%
\@twosidetrue
\@mparswitchtrue

%
% printindex
%
\newcommand*\see[2]{\emph{\seename} #1}
\newcommand\printindex{\@input@{\jobname.ind}}
\providecommand\seename{see}

%
% definitions for font size 12pt
%
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xiipt{14.5}%
   \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\newcommand\small{%
   \@setfontsize\small\@xipt{13.6}%
   \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 9\p@ \@plus3\p@ \@minus5\p@
               \parsep 4.5\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\footnotesize{%
   \@setfontsize\footnotesize\@xpt\@xiipt
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
\newcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
\newcommand\large{\@setfontsize\large\@xivpt{18}}
\newcommand\Large{\@setfontsize\Large\@xviipt{22}}
\newcommand\LARGE{\@setfontsize\LARGE\@xxpt{25}}
\newcommand\huge{\@setfontsize\huge\@xxvpt{30}}
\let\Huge=\huge
\setlength\parindent{1.5em}
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\setlength\headsep   {.275in}
\setlength\topskip   {12\p@}
\setlength\footskip{30\p@}
\setlength\maxdepth{.5\topskip}
\setlength\@tempdima{\paperwidth}
\addtolength\@tempdima{-2in}
\setlength\@tempdimb{390\p@}
\ifdim\@tempdima>\@tempdimb\relax
    \setlength\textwidth{\@tempdimb}
\else
    \setlength\textwidth{\@tempdima}
\fi
\@settopoint\textwidth
\setlength\@tempdima{\paperheight}
\addtolength\@tempdima{-2in}
\addtolength\@tempdima{-1.5in}
\divide\@tempdima\baselineskip
\@tempcnta=\@tempdima
\setlength\textheight{\@tempcnta\baselineskip}
\addtolength\textheight{\topskip}
\setlength\marginparsep{7\p@}
\setlength\marginparpush{7\p@}
\setlength\@tempdima        {\paperwidth}
\addtolength\@tempdima      {-\textwidth}
\setlength\oddsidemargin    {.4\@tempdima}
\addtolength\oddsidemargin  {-1in}
\setlength\marginparwidth   {.6\@tempdima}
\addtolength\marginparwidth {-\marginparsep}
\addtolength\marginparwidth {-0.4in}
\ifdim \marginparwidth >2in
    \setlength\marginparwidth{2in}
\fi
\@settopoint\oddsidemargin
\@settopoint\marginparwidth
\setlength\evensidemargin  {\paperwidth}
\addtolength\evensidemargin{-2in}
\addtolength\evensidemargin{-\textwidth}
\addtolength\evensidemargin{-\oddsidemargin}
\@settopoint\evensidemargin
\setlength\topmargin{\paperheight}
\addtolength\topmargin{-2in}
\addtolength\topmargin{-\headheight}
\addtolength\topmargin{-\headsep}
\addtolength\topmargin{-\textheight}
\addtolength\topmargin{-\footskip}     % this might be wrong!
\addtolength\topmargin{-.5\topmargin}
\@settopoint\topmargin
\setlength\footnotesep{8.4\p@}
\setlength{\skip\footins}{10.8\p@ \@plus 4\p@ \@minus 2\p@}
\setlength\floatsep    {12\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep   {14\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\dblfloatsep    {14\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{10\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{10\p@ \@plus 2fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}
\setlength\partopsep{3\p@ \@plus 2\p@ \@minus 2\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 5\p@  \@plus2.5\p@ \@minus\p@
            \topsep 10\p@ \@plus4\p@   \@minus6\p@
            \itemsep5\p@  \@plus2.5\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    5\p@   \@plus2.5\p@ \@minus\p@
              \parsep    2.5\p@ \@plus\p@    \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2.5\p@\@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}
%
% of end definitions for font size 12pt
%


\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}

\def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}}

\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }

\def\version#1{\gdef\@version{#1}}
\def\@version{}

\def\miniabstract#1{\gdef\@miniabstract{#1}}
\def\@miniabstract{}

\newcommand\maketitle{\pagestyle{empty}\begin{titlepage}
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}
    {\Huge\bf \@title \par}
    \vskip 1em
    \ifx\@version\@empty\else
    {\large\centerline{version \@version}\par}\fi
    \vskip 1em
    {\large \@date \par}
    \vskip 3em
    {\large
     \lineskip .75em
      \begin{tabular}[t]{c}
        \@author
      \end{tabular}\par}
      \vskip 1.5em
  \end{center}\par
  \ifx\@miniabstract\@empty\else{
  \vfill
  \noindent\@miniabstract
  }\fi
  \@thanks
  \vfil\null
  \end{titlepage}\cleardoublepage\pagestyle{headings}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\@version\@empty
  \global\let\@miniabstract\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \global\let\version\relax
  \global\let\miniabstract\relax
}

\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em
}

\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand\thepart         {\@Roman\c@part}
\renewcommand\thechapter      {\@arabic\c@chapter}
\renewcommand\thesection      {\thechapter.\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{\cleardoublepage
            \@mainmatterfalse\pagenumbering{roman}}
\newcommand\mainmatter{\cleardoublepage
       \@mainmattertrue\pagenumbering{arabic}}
\newcommand\backmatter{\cleardoublepage
      \@mainmatterfalse}
\newcommand\part{\cleardoublepage
                 \thispagestyle{plain}%
                 \null\vfil
                 \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
                \null
                \thispagestyle{empty}%
                \newpage
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\cleardoublepage
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                      \@makechapterhead{#2}%
                      \@afterheading
                    }
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{$\m@th\bullet$}
\newcommand\labelitemii{\normalfont\bfseries --}
\newcommand\labelitemiii{$\m@th\ast$}
\newcommand\labelitemiv{$\m@th\cdot$}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \newpage
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\newpage
    }
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \renewcommand\@chapapp{\appendixname}%
  \renewcommand\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset{equation}{chapter}
\renewcommand\theequation{\thechapter.\@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{2}
\newcommand\tableofcontents{%
    \cleardoublepage
    \pagenumbering{roman}
    \@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}
    \chapter*{\contentsname}
    \@starttoc{toc}
    \cleardoublepage
    \pagenumbering{arabic}
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \setlength\@tempdima{3em}%
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \@mkboth{\MakeUppercase\listfigurename}{\MakeUppercase\listfigurename}
    \chapter*{\listfigurename}
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \@mkboth{\MakeUppercase\listtablename}{\MakeUppercase\listtablename}
    \chapter*{\listtablename}
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}
      \chapter*{\bibname}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\onecolumn}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\newcommand\today{}
\edef\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
\onecolumn



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% special styles
%
\headheight 30pt
\footskip 36pt

\def\partmark#1{}
\def\chaptermark#1{}
\def\sectionmark#1{}

%
% names
%
\def\contentsname{Contents}
\def\listfigurename{List of Figures}
\def\listtablename{List of Tables}
\def\bibname{Bibliography}
\def\indexname{Index}
\def\figurename{Figure}
\def\tablename{Table}
\def\chaptername{Chapter}
\def\appendixname{Appendix}
\def\partname{Part}

%
% part definition
%
\def\part{\cleardoublepage \thispagestyle{plain} \if@twocolumn \onecolumn
\@tempswatrue \else \@tempswafalse \fi \hbox{}\vfil \secdef\@part\@spart}

\def\@part[#1]#2{\ifnum \c@secnumdepth >-2\relax \refstepcounter{part}
\addcontentsline{toc}{part}{\thepart \hspace{1em}#1}\else
\addcontentsline{toc}{part}{#1}\fi \markboth{}{}
\partmark{#1}
  \parbox{\textwidth}{
    \rule[3ex]{\textwidth}{0.5ex}
    {\mbox{}\hfill\Huge\bf\thepart~~\uppercase\expandafter{#2}\hfill\mbox{}}\\
    \rule{\textwidth}{0.5ex}
}
\@endpart}

\def\@endpart{\vfil\newpage \if@twoside \hbox{} \thispagestyle{empty}
 \newpage
 \fi \if@tempswa \twocolumn \fi}

\def\@spart#1{{\centering \Huge \bf #1\par}\@endpart}


%
% chapter definition
%
\def\@makechapterhead#1{
  \vspace*{30pt}
  { \parindent 0pt \raggedright
    \rule[3ex]{\textwidth}{0.5ex}
    \parbox{\textwidth}{\Huge\bf \thechapter~\uppercase\expandafter{#1}}
%  \makebox[0.1\columnwidth][l]{\Huge\bf\thechapter}\begin{minipage}[t]{0.899\columnwidth}\Huge\bf\uppercase\expandafter{#1}\end{minipage}
  \nobreak \vskip 30pt } }
\def\@makeschapterhead#1{
  \vspace*{30pt}
  { \parindent 0pt \raggedright
    \rule[3ex]{\textwidth}{0.5ex}
    \parbox{\textwidth}{\Huge\bf\uppercase\expandafter{#1}}
  \nobreak \vskip 30pt } }



\def\chapter{\cleardoublepage \thispagestyle{plain} \global\@topnum\z@
\@afterindentfalse \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
 \refstepcounter{chapter}
 \typeout{\@chapapp\space\thechapter.}
 \addcontentsline{toc}{chapter}{\protect
 \numberline{\thechapter}#1}\else
 \addcontentsline{toc}{chapter}{#1}\fi
 \chaptermark{#1}
 \addtocontents{lof}{\protect\addvspace{10pt}}
\addtocontents{lot}{\protect\addvspace{10pt}} \if@twocolumn
\@topnewpage[\@makechapterhead{#2}]
 \else \@makechapterhead{#2}
 \@afterheading \fi}

\def\@schapter#1{\if@twocolumn \@topnewpage[\@makeschapterhead{#1}]
 \else \@makeschapterhead{#1}
 \@afterheading\fi}
 


%
% section definition
%
%
%\def\@sect#1#2#3#4#5#6[#7]#8{\ifnum #2>\c@secnumdepth
%     \def\@svsec{}\else
%     \refstepcounter{#1}\edef\@svsec{\csname the#1\endcsname\ \ }\fi
%     \@tempskipa #5\relax
%      \ifdim \@tempskipa>\z@
%        \begingroup #6\relax
%          \relax\@svsec{\interlinepenalty \@M
%          \uppercase\expandafter{#8}\par}%
%        \endgroup
%       \csname #1mark\endcsname{#7}\addcontentsline
%         {toc}{#1}{\ifnum #2>\c@secnumdepth \else
%                      \protect\numberline{\csname the#1\endcsname}\fi
%                    #7}\else
%        \def\@svsechd{#6 \@svsec
%          \uppercase\expandafter{#8}\csname #1mark\endcsname
%                      {#7}\addcontentsline
%                           {toc}{#1}{\ifnum #2>\c@secnumdepth \else
%                             \protect\numberline{\csname the#1\endcsname}\fi
%                       #7}}\fi
%     \@xsect{#5}}

%\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
%   \ifdim \@tempskipa>\z@
%     \begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M
%     \uppercase\expandafter{#5}\par}\endgroup
%   \else \def\@svsechd{#4\hskip #1\relax
%     \uppercase\expandafter{#5}}\fi
%   \@xsect{#3}}






%\def\section{\@startsection {section}{1}{0pt}{-3.5ex plus -1ex minus
% -.2ex}{2.3ex plus .2ex}{
%\noindent\parbox{\textwidth}{\rule{\textwidth}{0.2ex}}
%\noindent\Large\bf}}

%\def\subsection{\@startsection{subsection}{2}{\z@}{-3.25ex plus -1ex minus
% -.2ex}{1.5ex plus .2ex}{
%\noindent\parbox{\textwidth}{\rule{\textwidth}{0.1ex}}
%\noindent\large\bf}}

%\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-3.25ex plus
% -1ex minus -.2ex}{1.5ex plus .2ex}{
%\noindent\parbox{\textwidth}{\rule{\textwidth}{0.1ex}}
%\noindent\normalsize\bf}}

%\def\paragraph{\@startsection{paragraph}{4}{\z@}{-3.25ex plus
% -1ex minus -.2ex}{1.5ex plus .2ex}{
%\noindent\normalsize\bf}}

%\def\subparagraph{\@startsection{subparagraph}{5}{\z@}{-3.25ex plus
% -1ex minus -.2ex}{1.5ex plus .2ex}{
%\noindent\normalsize\bf}}






%
% pagestyle plain
%
\def\ps@plain{\let\@mkboth\@gobbletwo
     \def\@oddhead{}\def\@oddfoot{\rm\hfil{\large\bf\thepage}
     \hfil}\def\@evenhead{}\let\@evenfoot\@oddfoot}


%
% pagestyle headings
%
\def\ps@headings{\let\@mkboth\markboth
\def\@oddfoot{}
\def\@evenfoot{}
\def\@evenhead{\parbox{\textwidth}{
\rule{\textwidth}{0.1ex}
{\large\bf\thepage}~~~~~{\small\leftmark}\\
\rule[1.45ex]{\textwidth}{0.1ex}}}
\def\@oddhead{\parbox{\textwidth}{
\rule{\textwidth}{0.1ex}
\hbox{}~\hfill {\small\rightmark}~~~~~{\large\bf\thepage}\\
\rule[1.45ex]{\textwidth}{0.1ex}}}
\def\partmark##1{\markboth{\uppercase\expandafter{##1}}{}}
\def\chaptermark##1{\markboth{\thechapter.\ \uppercase\expandafter{##1}}{}}
\def\sectionmark##1{\markright{\thesection.\ \uppercase\expandafter{##1}}}}


%
% default page settings
%
\ps@headings
\pagenumbering{arabic}
\onecolumn


%
% chapter, section, subsection, subsubsection`
%
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

%
% floating objects, vertical spaces etc.
%
\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\floatpagefraction}{1.0}

%
% page sizes
%
\sloppy

\textheight 23cm
\topmargin -0.54cm
\textwidth 16.5cm
\oddsidemargin -0.04cm
\evensidemargin -0.54cm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% special floating objects
%
%
% floating object 'Algorithm'
%
\def\listalgorithmname{List of Algorithms}
\def\algorithmname{Algorithm}
\let\l@algorithm\l@figure
\newcounter{algorithm}[chapter]
\def\thealgorithm{\thechapter.\@arabic\c@algorithm}
\def\fps@algorithm{htb}
\def\ftype@algorithm{3}
\def\ext@algorithm{loa}
\def\fnum@algorithm{\algorithmname~\thealgorithm}
\def\algorithm{\@float{algorithm}}
\let\endalgorithm\end@float
\@namedef{algorithm*}{\@dblfloat{algorithm}}
\@namedef{endalgorithm*}{\end@dblfloat}
\def\listofalgorithms{
    \@mkboth{\MakeUppercase\listalgorithmname}{\MakeUppercase\listalgorithmname}
    \chapter*{\listalgorithmname}
    \@starttoc{\ext@algorithm}
    }


%
% floating object 'Example'
%
\def\listexamplename{List of Examples}
\def\examplename{Example}
\let\l@example\l@figure
\newcounter{example}[chapter]
\def\theexample{\thechapter.\@arabic\c@example}
\def\fps@example{htb}
\def\ftype@example{4}
\def\ext@example{loe}
\def\fnum@example{\examplename~\theexample}
\def\example{\@float{example}}
\let\endexample\end@float
\@namedef{example*}{\@dblfloat{example}}
\@namedef{endexample*}{\end@dblfloat}
\def\listofexamples{
    \@mkboth{\MakeUppercase\listexamplename}{\MakeUppercase\listexamplename}
    \chapter*{\listexamplename}
    \@starttoc{\ext@example}
    }


%
% floating object 'Program'
%
\def\listprogramname{List of Programs}
\def\programname{Program}
\let\l@program\l@figure
\ifx\thechapter\@undefined
    \newcounter{program}
    \def\theprogram{\@arabic\c@program}
\else
    \newcounter{program}[chapter]
    \def\theprogram{\thechapter.\@arabic\c@program}
\fi
\def\fps@program{htb}
\def\ftype@program{5}
\def\ext@program{lop}
\def\fnum@program{\programname~\theprogram}
\def\program{\@float{program}}
\let\endprogram\end@float
\@namedef{program*}{\@dblfloat{program}}
\@namedef{endprogram*}{\end@dblfloat}
\def\listofprograms{
    \@mkboth{\MakeUppercase\listprogramname}{\MakeUppercase\listprogramname}
    \chapter*{\listprogramname}
    \@starttoc{\ext@program}
    }


\endinput
