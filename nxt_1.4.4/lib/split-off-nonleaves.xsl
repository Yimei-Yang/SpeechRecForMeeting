<?xml version="1.0" encoding="ISO-8859-1"?>
<!-- SPLIT-0FF-NONLEAVES.XSL           JEAN CARLETTA        28 JAN03 

This stylesheet together with split-off-leaves implements a 
generic unknitting that takes the leaves
of an xml file, whatever they are, and throws them into a separate
file, with the appropriate LTXML1 style link attributes in the NITE
data format (i.e., using nite:child elements) added to the elements 
that used to contain the leaves.  Usage:

java org.apache.xalan.xslt.Process -in INFILE -xsl split-off-leaves.xsl
-param doctype DOCTYPE -param docid DOCID

where DOCTYPE names the element to be used at the root of the bottom
document, and DOCID names the id of that element (we like to have ids
on everything).

java org.apache.xalan.xslt.Process -in INFILE -xsl split-off-nonleaves.xsl
-param leaffile FILE

where FILE names the file where we put the leaves

Note that this is implemented so that including a stylesheet that defines
a different link syntax will work without changes to this one.  We think.
And with the exception that if one doesn't have the namespace declarations
for the links on this stylesheet element, then they get placed on the 
individual elements that need them in the output, which looks bad (but
probably works).

This should have parameters for the id attribute name and child and pointer
element names to use.  
-->

<xsl:stylesheet 
   xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
   xmlns:nite="http://nite.sourceforge.net/"
   version="1.0">
   

  <xsl:output method="xml" indent="yes" encoding="ISO-8859-1"/>
  <!-- stripping space is not safe with mixed content children - beware -->
  <xsl:strip-space elements="*"/>
  <xsl:include href="ltxml1-link-templates.xsl"/>

  <xsl:param name="leaffile"/>

  <xsl:template match="/">
    <xsl:apply-templates/>      
  </xsl:template>

<!--The following two templates copy the input to the output for
     anything that has a child.

     There is a textual difference, though; defaulted attributes
     are written in the output. There is probably a test for this
     in XSL, so we might be able to change that.  It would make the
     selection attribute more complex.


-->
<xsl:template match="@*">
  <xsl:copy/>
</xsl:template>

<xsl:template match="*[child::*]" priority="2">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
</xsl:template>


<!-- match on anything left (which is just the leaves) and do nothing  -->
<xsl:template match="*" priority="1">
      <xsl:call-template name="childmaker">
        <xsl:with-param name="filename" select="$leaffile"/>
        <xsl:with-param name="firstid" select="@nite:id"/>
        <xsl:with-param name="secondid" 
               select="@nite:id"/>
      </xsl:call-template>
</xsl:template>

</xsl:stylesheet>
