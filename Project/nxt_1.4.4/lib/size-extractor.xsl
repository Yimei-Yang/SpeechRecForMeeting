<xsl:stylesheet version='1.0' 
                xmlns:xsl='http://www.w3.org/1999/XSL/Transform' 
                xmlns:xlink='http://www.w3.org/1999/xlink'
>
<!--    SIZE-EXTRACTOR.XSL   JEAN CARLETTA  02 JULY 04
	    
Assuming output from SaveQueryResults, just print out an indented
tree of the size attributes on the matchlist attributes, giving
a more hierarchical count breakdown than CountQueryResults will
give.

Call as:

     java org.apache.xalan.xslt.Process -in INFILE -xsl  size-extractor.xsl -out OUTFILE

The only problem I can see is that there are extra blank lines
in the output that we don't want.  
-->  


<xsl:output method="text" indent="yes" encoding="ISO-8859-1" /> 
<xsl:strip-space elements="*"/>
<xsl:preserve-space elements="matchlist"/>

<xsl:template match="/">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="matchlist">
   <xsl:value-of select="@size"/>
   <xsl:apply-templates/>
</xsl:template>


</xsl:stylesheet>
