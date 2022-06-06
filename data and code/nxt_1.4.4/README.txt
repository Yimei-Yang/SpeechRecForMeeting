The NITE XML Toolkit Version 1.4.4

-------------
PREREQUISITES
-------------
Java 1.4.2_05 or above

------------
INSTALLATION
------------
To install the software download nxt.zip and unzip it. Provided you
have at least Java 1.4.2_05 installed you should find you can run the
.bat, .sh or .command files for PC UNIX and Mac respectively, that are
included with the distribution.  For example, execute 'sh smartkom.sh'
on a UNIX system

---------------------------------------------------------
BUILDING FROM SOURCE (requires ant + Java 1.5 or greater)
---------------------------------------------------------
Download the source distribution: nxt-src.zip and unzip. Copy the
build.xml script from the 'build_scripts' directory into the 'nxt'
directory and from there type 'ant'. You should find this builds a new
'nxt.jar' file in the 'lib' directory, and you should be able to
execute the .sh, .bat or .command file named 'search'.

In order to build successfully, you may also need to download
(via the CVS repository) part or all of the 'lib' directory,
depending on what you have installed and on your CLASSPATH.

-----------
INFORMATION
-----------
This software, which is written in Java and distributed under the
GNU General Public License, provides integrated
implementations of:

# the NXT Model, a data model designed for multimodal corpora
 with linguistic annotation that supports multi-rooted trees with
 some arbitrary graph structure imposed over the top;

# the Nite Query Language (NQL), exploiting the full range of
 flexiblity and expressive power in the NITE Object Model;

# the NITE Display Objects (NDO), a library of Java display
objects which can be used to build interfaces based on either 
NXT Model or standard JDOM handling;

# the NITE Stylesheet Engine (NSE), which, given a stylesheet 
that declaratively specifies the mapping between the data and
 the display, builds it.

In addition to these library components, the download contains
Javadoc generated from the source, sample data, sample
 programs, and jar files for the libraries we use.

------------------------------
EXTERNAL LIBRARIES USED BY NXT
------------------------------
Each external library used by NXT has its own LICENCE file in the
'lib/licences' directory of this distribution.

file               version   web page
-------------------------------------------------------------------------------
xercesImpl.jar     2.10.0    http://xerces.apache.org/
xml-apis.jar       2.10.0    [collection of XML APIs used in the Xerces project]
xalan.jar          2.7.0     http://xml.apache.org/xalan-j/index.html
jh.jar             2.0_05    http://java.sun.com/products/javahelp/
JMF directory      2.1.1     http://java.sun.com/products/java-media/jmf/
pnuts.jar          1.0       https://pnuts.dev.java.net/
  -- later versions of PNuts no longer have the layout functionality used 
     by some old NXT code, so westay on 1.0.
jmanual.jar, helpset.jar     http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERSearch/subprojects.shtml
poi.jar            3.6       http://poi.apache.org/
forms.jar          1.3.0     http://www.jgoodies.com/
looks.jar          2.3.1     http://www.jgoodies.com/
eclipseicons.jar   3.1?      http://www.eclipse.org/ [ icons only under EPL ]
prefuse.jar        beta      http://prefuse.org/     [ 21.10.2007 beta release ]

fmj/fmj.jar        patched original code from FMJ CVS repository, 20/4/07,
patched by Craig Nicol to use Swing rather than AWT component (see lib/fmj/Handler.java for the patch). See http://fmj.sourceforge.net/
fmj/lib/*.jar      fmj-20061212-1055 http://fmj.sourceforge.net/ 
All FMJ jar files are covered by the licence lib/licences/00LICENSE_FMJ.txt

JavaCC version 2.0 [https://javacc.dev.java.net/] was used to compile
the NQL parser but is not included here (newest version is now 5.0)

This distribution contains corpus data that is copyright HCRC (see
Data/xml/MapTask/MAPTASK_LICENCE).

All other software and data within this distribution is copyright (c)
2003 HCRC, Edinburgh / IMS, Stuttgart and is distributed under the
terms of the GNU public licence (http://www.gnu.org/copyleft/gpl.html).

For more information about the software and the XML developments
in the NITE project, see http://www.ltg.ed.ac.uk/NITE.

--------------------------
CHANGES SINCE LAST VERSION
--------------------------

Active developers since last version: - Jonathan Kilgour

GENERAL
-------

This is a minor release addressing some bugs / feature requests.


NEW FEATURES
------------


SOURCEFORGE BUG FIXES
---------------------

[3061884] Save can cause loss of child elements
[3061865] getParents can fail to return all parents
[2972725] Corpus Help sometimes fails to manifest
[2904915] Iteratively loading >1 obs. fails w/ forceResourceLoad


SOURCEFORGE FEATURE REQUESTS
----------------------------

[2933005] Allow environment variables where java properties are currently used
