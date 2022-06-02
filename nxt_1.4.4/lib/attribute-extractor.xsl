<xsl:stylesheet version='1.0' 
                xmlns:xsl='http://www.w3.org/1999/XSL/Transform' 
                xmlns:xlink='http://www.w3.org/1999/xlink'
>
<!--    DURATION-EXTRACTOR.XSL   JEAN CARLETTA  02 JULY 04
	    
Assuming output from SaveQueryResults that has been knitted with
the data set on which it was run and where the  variable with the
name VARNAME has been bound to something with an attribute named
ATTNAME, extracts a flat list of the attribute values for matches
to that variable, as plain text.

Call as:

     java org.apache.xalan.xslt.Process -in INFILE -xsl  duration-extractor.xsl -out OUTFILE -param attribute ATTNAME -param variable VARNAME

Not all stylesheet processors can handle parameters passed in, and
this seems somewhat brittle, so if it doesn't work the first thing to
try is hard-wiring the values (see xsl:param calls at start below).

We default to an attname of "duration" and a variable name of "dur",
just because we might as well default to something to show what to
do for stylesheet processors that can't handle parameters, 
and that's what the first user wanted.
-->  

<!-- CHANGE COMMAND LINE PARAMETER DEFAULTS HERE -->
<xsl:param name="attname">duration</xsl:param>
<xsl:param name="varname">dur</xsl:param>

<xsl:output method="text" indent="no" encoding="ISO-8859-1" /> 

<xsl:template match="/">
  <xsl:apply-templates select="//*[name(.)=concat('match-',$varname)]"/>
</xsl:template>

<!-- Find the elements that have the name match-VARNAME -->
<xsl:template match="*">
   <!-- whatever type the child is, output just the attribute 
        named ATTNAME. There should always be just one child,
        so we aren't careful about how this is done -->
   <xsl:value-of select="child::*/@*[name(.)=$attname]"/>
   <!-- this inserts a new line.  There's some better way
        of specifying this using character codes like &#xD;&#xA;
        but I can't remember what the best sequence is. -->
   <xsl:text>
</xsl:text>
</xsl:template>


</xsl:stylesheet>
