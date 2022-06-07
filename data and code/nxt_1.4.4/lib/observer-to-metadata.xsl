<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:strip-space elements="*"/>
<xsl:output indent="yes"/>

<!-- Transform the XML configuration file from Henk Roozen's XML
  output module from the Observer into NITE metadata format. Note that
  observations do not appear in the config file, so must be gathered
  separately (a program to do this "Observer2Nite" is available)

  Jonathan Kilgour 3/3/3
-->

<xsl:template match="/">
<xsl:element name="corpus">
<xsl:attribute name="id"><xsl:value-of select="corpus/@id"/></xsl:attribute>
<xsl:attribute name="description"><xsl:value-of select="corpus/@description"/></xsl:attribute>
<!-- Henk always uses 'stream' as the stream element name! -->
<reserved-elements>
  <stream name="stream"/>
</reserved-elements>
<xsl:apply-templates select="corpus/independentvariablesgroup"/>
<xsl:apply-templates select="corpus/subjectsgroup"/>
<xsl:apply-templates select="corpus/behaviorsgroup"/>
</xsl:element>
</xsl:template>

<xsl:template match="subjectsgroup">
<agents>
  <xsl:apply-templates select="subject"/>
</agents>
</xsl:template>

<xsl:template match="subject">
<xsl:element name="agent">
<!-- <xsl:attribute name="name"><xsl:apply-templates select="code"/></xsl:attribute> -->
<xsl:attribute name="name"><xsl:value-of select="translate(normalize-space(name/text()), ' ', '_')"/></xsl:attribute>
<xsl:attribute name="description"><xsl:value-of select="normalize-space(name/text())"/></xsl:attribute>
</xsl:element>
</xsl:template>

<xsl:template match="independentvariablesgroup">
<observation-variables>
  <xsl:apply-templates select="independentvariable"/>
</observation-variables>
 </xsl:template>

<xsl:template match="independentvariable">
<xsl:element name="observation-variable">
<xsl:attribute name="name"><xsl:value-of select="normalize-space(name/text())"/></xsl:attribute>
<xsl:variable name="tyvar" select="normalize-space(type/text())"/>
<xsl:attribute name="type">
 <xsl:choose>
		<xsl:when test="$tyvar='2'">number</xsl:when>
		<xsl:otherwise>string</xsl:otherwise>
</xsl:choose>
</xsl:attribute>
</xsl:element>
</xsl:template>

<xsl:template match="behaviorsgroup">
<codings path=".">
<agent-codings><xsl:apply-templates select="behaviorclass"/></agent-codings>  
</codings>
</xsl:template>

<xsl:template match="behaviorclass">
<xsl:element name="coding-file">
<xsl:variable name="tyvar" select="normalize-space(type/text())"/>
<xsl:attribute name="name"><xsl:value-of select="translate(normalize-space(name/text()),' ','_')"/></xsl:attribute>
<xsl:element name="time-aligned-layer"><xsl:attribute name="name"><xsl:value-of select="normalize-space(name/text())"/>-layer</xsl:attribute><xsl:apply-templates select="behavior"/>
</xsl:element>
</xsl:element>
</xsl:template>

<xsl:template match="behavior">
<xsl:element name="code">
<!-- <xsl:attribute name="name"><xsl:value-of select="translate(normalize-space(code/text()),' ','_')"/></xsl:attribute> -->
<xsl:attribute name="name"><xsl:value-of select="translate(normalize-space(name/text()),' ','_')"/></xsl:attribute>
<xsl:apply-templates select="modifier1|modifier2|modifier3|modifier4"/>
<!-- any element can have a 'comment' attribute! -->
<attribute name="comment" type="string"/>
</xsl:element>
</xsl:template>

<xsl:template match="modifier1|modifier2|modifier3|modifier4">
 <xsl:variable name="mt" select="normalize-space(./text())"/>
 <xsl:variable name="mod" select="/corpus/modifiersgroup/modifierclass[normalize-space(name/text())=$mt]"/>
 <xsl:element name="attribute" >
  <xsl:attribute name="name"><xsl:value-of select="$mt"/></xsl:attribute> 
<!--  <xsl:attribute name="name"><xsl:value-of select="name(.)"/></xsl:attribute> -->
<xsl:attribute name="type">
 <xsl:choose>
		<xsl:when test="$mod/type/text()='2'">number</xsl:when>
		<xsl:when test="$mod/modifier">enumeration</xsl:when>
		<xsl:otherwise>string</xsl:otherwise>
</xsl:choose>
</xsl:attribute>
<xsl:apply-templates select="$mod/modifier"/>
</xsl:element>
</xsl:template>

<xsl:template match="modifier">
<!-- <value><xsl:value-of select="normalize-space(code/text())"/></value> -->
<value><xsl:value-of select="translate(normalize-space(name/text()),' ','_')"/></value>
</xsl:template>

</xsl:stylesheet>
