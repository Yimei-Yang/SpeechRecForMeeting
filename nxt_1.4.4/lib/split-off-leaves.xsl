<?xml version="1.0" encoding="ISO-8859-1"?>
<!-- SPLIT-0FF-LEAVES.XSL           JEAN CARLETTA        28 JAN03 

This stylesheet together with split-off-nonleaves implements a 
generic unknitting that takes the leaves
of an xml file, whatever they are, and throws them into a separate
file, with the appropriate LTXML1 style link attributes added to the
elements that used to contain the leaves.  Usage:

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
  
  <xsl:param name="doctype"/>
  <xsl:param name="docid"/>

  <xsl:template match="/">
      <xsl:element name="{$doctype}">
        <xsl:attribute name="nite:id">
          <xsl:value-of select="$docid"/>
        </xsl:attribute>
        <!-- there's probably a way to go straight to leaves, but
             I'm not sure how.  It's not select="*[not(child::*)]"  -->
        <xsl:apply-templates/>
      </xsl:element>
  </xsl:template>

<xsl:template match="@*">
  <xsl:copy/>
</xsl:template>


<!-- suppress output for elements that have a child element -->

<xsl:template match="*[child::*]" priority="2">
  <xsl:apply-templates/>
</xsl:template>

<!-- copy any element that's left (which is just the leaves) -->
<xsl:template match="*" priority="1">
   <xsl:copy><xsl:apply-templates select="@*|node()"/></xsl:copy>
</xsl:template>

</xsl:stylesheet>
