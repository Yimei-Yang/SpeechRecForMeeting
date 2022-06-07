<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
  xmlns:lxslt="http://xml.apache.org/xslt"
  xmlns:nite="http://nite.sourceforge.net/">
  
 <xsl:strip-space elements="*"/>
 <xsl:output method="xml" indent="yes"/>

 <!-- JONATHAN KILGOUR

   Take the transcript for the AMI corpus and break it into one
   file per speaker so we can use it with the audio.

   The awful XML that comes out of channeltrans looks like this:
    <Sync chan="default" time="367.909"/>
    Uh-huh.
    <Sync chan="2" time="255.115"/>
    Right, okay.
 
    So you have to assume the line following a Sync belongs to a particular
    channel rather than using elements to contain the text. It
    seems that all utterances from a speaker are together in a file, so
    perhaps that's a way of splitting?

 -->

 <xsl:param name="session"/>
 <xsl:param name="participant"/>
      
 <!-- this copies all codes adding an extra attribute called "dial" that contains
     the dialogue id passed in -->

 <!-- at the document root, make a nite:root element -->
 <xsl:template match="/">
   <xsl:element name="nite:root" xmlns:nite="http://nite.sourceforge.net/">
     <xsl:attribute name="nite:id">
       <xsl:value-of select="concat(concat($session,'.'),$participant)"/>
     </xsl:attribute>
<!-- want to select all turns that have segs with chan attribute=participant -->	
	<xsl:apply-templates select="//sync[@chan=$participant]"/>
<!--     <xsl:apply-templates select="//turn"/> -->
   </xsl:element>
 </xsl:template>


 <!-- for everything else, copy the element -->
<xsl:template match="sync" mode="following">
</xsl:template>

 <!-- for everything else, copy the element -->
<xsl:template match="sync">
  <xsl:copy>
   <xsl:apply-templates select="@*"/>	
   <xsl:apply-templates select="node()"/>	
   <xsl:apply-templates select="following-sibling::node()[1]" mode="following"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="text()" mode="following">
  <xsl:copy>
   <xsl:apply-templates select="node()"/>	
  </xsl:copy>
</xsl:template>

<xsl:template match="@*">
  <xsl:copy/>
</xsl:template>

</xsl:stylesheet>
