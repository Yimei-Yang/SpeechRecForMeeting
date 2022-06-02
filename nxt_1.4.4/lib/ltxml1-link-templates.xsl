<?xml version="1.0" encoding="ISO-8859-1"?>
<!-- LTXML1-LINK-TEMPLATES.XSL           JEAN CARLETTA        28 JAN03 

Common xsl templates for creating data in the NITE NXT stand-off
data format using LTXML1 link style.
-->

<xsl:stylesheet 
   xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
   xmlns:nite="http://nite.sourceforge.net/"
   version="1.0">

<!-- this is a generic named template for creating the href attribute
     required for a child link. 
     Should allow null secondid to mean it's a single element reference.
     Should also allow one to pass the name of the element to use for
     child/pointer, since that's configurable.
-->
<xsl:template name="childmaker">
  <xsl:param name="filename"/>
  <xsl:param name="firstid"/>
  <xsl:param name="secondid"/>
  <xsl:element name="nite:child">
    <xsl:attribute name="href">
      <xsl:choose>
        <xsl:when test="$firstid != $secondid">
          <xsl:value-of select="concat($filename, '#id(', $firstid, ')..id(', $secondid, ')')"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="concat($filename, '#id(', $firstid,')')"/>     
        </xsl:otherwise>
      </xsl:choose>
    </xsl:attribute>
  </xsl:element>
</xsl:template>

<!-- same but for nite pointers -->
<xsl:template name="pointermaker">
  <xsl:param name="filename"/>
  <xsl:param name="firstid"/>
  <xsl:param name="secondid"/>
  <xsl:param name="role"/>
  <xsl:element name="nite:pointer">
    <xsl:attribute name="role">
      <xsl:value-of select="$role"/>
    </xsl:attribute>
    <xsl:attribute name="href">
      <xsl:choose>
        <xsl:when test="$firstid != $secondid">
          <xsl:value-of select="concat($filename, '#id(', $firstid, ')..id(', $secondid, ')')"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="concat($filename, '#id(', $firstid,')')"/>     
        </xsl:otherwise>
      </xsl:choose>
    </xsl:attribute>
  </xsl:element>
</xsl:template>

<!-- we also thought it might be useful to have a stylesheet
     for building stream elements.  At the moment, however,
     they only have ids.  I thought they might need more 
     things like namespace declarations specifically listed here,
     but in the current stylesheets using the main output, 
     these get added automatically.
-->
<xsl:template name="makestream-attrs">
    <xsl:param name="id"/>
      <xsl:attribute name="nite:id">
        <xsl:value-of select="$id"/>
      </xsl:attribute>
</xsl:template>


</xsl:stylesheet>
