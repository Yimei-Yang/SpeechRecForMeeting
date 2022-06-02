<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
  xmlns:lxslt="http://xml.apache.org/xslt"
  xmlns:nite="http://nite.sourceforge.net/">
  
 <xsl:output method="xml" indent="yes"/>

<xsl:template match="nite:DisplayObject">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:element name="xsl:attribute">
	<xsl:attribute name="name">nitesourceid</xsl:attribute> 
        <xsl:element name="xsl:value-of">
 	  <xsl:attribute name="select">@id</xsl:attribute> 
        </xsl:element>
    </xsl:element>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="*">
  <xsl:copy>
   <xsl:apply-templates select="@*|node()"/>	
  </xsl:copy>
</xsl:template>

<xsl:template match="@*">
  <xsl:copy/>
</xsl:template>

</xsl:stylesheet>
