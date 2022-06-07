<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:strip-space elements="*"/>
<xsl:output indent="yes"/>

<!-- This stylesheet copies the input, but takes modifier elements and
  makes them into attributes on the parent element. For example

 <walking nite:id="a2" nite:start="27.650" nite:end="30.590">
  <modifier index="1" name="direction" value="south"/>
 </walking>

 BECOMES
 
 <walking nite:id="a2" nite:start="27.650" nite:end="30.590" direction="south"/>

 The reason this is necessary is that Observer output can have
 repeated modifiers i.e. more than one modifier with the same name on
 the same element. We simply want to give an error in this case
 because NITE cannot cope with that.

 Jonathan Kilgour 7/3/3
 -->

<xsl:template match="/">
<xsl:apply-templates/>
</xsl:template>

<xsl:template match="*">
  <xsl:choose>
    <xsl:when test="name(.)='modifier'">
	<xsl:attribute name="{@name}"><xsl:value-of select="@value"/></xsl:attribute>
    </xsl:when>
    <xsl:otherwise>
     <xsl:copy>
       <xsl:copy-of select="@*"/>
       <xsl:apply-templates/>
     </xsl:copy>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match="@">
 <xsl:copy-of select="."/>
</xsl:template>


</xsl:stylesheet>
