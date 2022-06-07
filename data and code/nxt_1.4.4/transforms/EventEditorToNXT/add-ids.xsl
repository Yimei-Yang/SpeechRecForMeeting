<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
  xmlns:lxslt="http://xml.apache.org/xslt"
  xmlns:nite="http://nite.sourceforge.net/">
  
 <xsl:output method="xml" indent="yes"/>

 <!-- JONATHAN KILGOUR   add-ids.xsl
    This stylesheet is based on preprocess-for-NXT by JEAN CARLETTA
    It simply adds ids to everything   -->

 <xsl:param name="session"/>
 <xsl:param name="participant"/>

<!-- stick an nite:root element at the root so we get the nite
  namespace declared 
 <xsl:template match="/">
   <xsl:element name="nite:root" xmlns:nite="http://nite.sourceforge.net/">
     <xsl:attribute name="nite:id">
       <xsl:value-of select="concat(concat($session,'.'),$participant)"/>
     </xsl:attribute>
     <xsl:apply-templates/>
   </xsl:element>
 </xsl:template>
-->

 <!-- for everything, copy the element, adding an id attribute -->
<xsl:template match="*">
  <xsl:copy>
   <xsl:apply-templates select="@*"/>	
   <xsl:attribute name="nite:id" xmlns:nite="http://nite.sourceforge.net/">
     <xsl:variable name="name">
       <xsl:value-of select="name(.)"/>
     </xsl:variable>
     <xsl:variable name="id">
       <xsl:number level="any"/>
     </xsl:variable>
     <xsl:value-of select="concat(concat(concat(concat(concat($session,'.'),$participant),$name),'.'),$id)"/>
   </xsl:attribute>
   <!-- might as well add durations to segments while we're here 
        decided against this - the iaccurate maths looks sloppy!
   <xsl:if test="name(.)='Segment'">
   <xsl:attribute name="dur">
     <xsl:value-of select="@EndTime - @StartTime"/>
   </xsl:attribute>
   </xsl:if>
   -->
   <xsl:apply-templates select="node()"/>	
  </xsl:copy>
</xsl:template>

<xsl:template match="@*">
  <xsl:copy/>
</xsl:template>

</xsl:stylesheet>
