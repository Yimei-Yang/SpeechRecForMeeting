<xsl:stylesheet version='1.0' 
                xmlns:xsl='http://www.w3.org/1999/XSL/Transform' 
                xmlns:xlink='http://www.w3.org/1999/xlink'
>
<!-- xmlns:nite="http://nite.sourceforge.net/" -->

<!--    KNIT.XSL - a generic knit facility for either NXT link style.
	    
Use this stylesheet to "knit" an NXT format XML file with data it
points to in other documents via out-of-file child and pointer links.
Stylesheet processors expect it to be in the same directory as the xml
file it is called on.

This version of the stylesheet is for corpora that use either the
LTXML syntax or a limited version of xpointer syntax; the choice
should be passed in at the command line using a -param linkstyle ltxml
or -param linkstyle xpointer.  The default is ltxml format.
	
In NXT format, officially one can use any element names one wishes for
out-of-file children and pointers; nite:child and nite:pointer are
simply the defaults.  Similarly, the ids used in the href references
can use any attribute, with nite:id as the default.  The choice has to
be consistent for an entire corpus.  When NXT is loading data, we know
what choices the corpus designer made by reading the metadata file.
Knit doesn't use the metadata, so instead we pass in at the command
line the element and attribute names.  Suppose some corpus were to use
the normal names but not namespace them.  The call for the stylesheet
might then be
 
     java org.apache.xalan.xslt.Process -in INFILE -xsl knit.ltxml.xsl -param idatt id -param childel child -param pointerel pointer -param linkstyle ltxml -param docbase file:///my/file/directory 2> errlog > OUTFILE

Not all stylesheet processors can handle parameters passed in, and
this seems somewhat brittle, so if it doesn't work the first thing to
try is hard-wiring the values (see xsl:param calls at start below).

The docbase parameter indicates the directory of the INFILE, used to
resolve the relative paths in child and pointer links. If not
specified, it will default to the location of the stylesheet (NOT the
input file!!!). Note that if you're using the absolute location of the
INFILE, it is perfectly fine to just set docbase to the same thing,
because the entity resolver will take its base URL (according to xslt
standard) for document function calls.

For out-of-file children, knit inserts a copy of the child as a normal
XML child.  That's uncontroversial.  It's less clear what behaviour
should be used for pointers, since these are allowed to introduce
cycles.  This stylesheet goes ahead and inserts the target node, but
doesn't trace its children, in order to avoid loops.  Between the
element pointed to and the element that points, it inserts a new
element with a name derived from the parent name and pointer role,
separated by a hyphen.  This element has ref and pointer attributes;
ref is the original href, and pointer is "yes", helping to identify
them in later processing.  This process could end up with bad XML if an
element ends up in the file twice, once because it can be found via a
child and once because it can be found via a pointer.  Also, nothing
about NXT format guarantees unique ids in the corpus as a whole, so
the ids may not not be unique in the knitted file (which is only a
problem if you write a DTD identifying them as ids).

We expect to distribute a faster, non-stylesheet knitting process that
will split off utilities for child and pointer inclusion, so that both
need to be run on some data in order to get the effect of this
stylesheet.

-->  

<!-- CHANGE COMMAND LINE PARAMETER DEFAULTS HERE -->
<xsl:param name="idatt">nite:id</xsl:param>
<xsl:param name="childel">nite:child</xsl:param>
<xsl:param name="pointerel">nite:pointer</xsl:param>
<xsl:param name="linkstyle">ltxml</xsl:param>
<xsl:param name="docbase"></xsl:param>

  <xsl:variable name="xptrlinkprefix">
    <xsl:text>#xpointer(id('</xsl:text>
  </xsl:variable>
  <xsl:variable name="xptrlinksuffix">
    <xsl:text>')</xsl:text>
  </xsl:variable>
  <xsl:variable name="xptrrangeprefix">
    <xsl:text>/range-to(id('</xsl:text>
  </xsl:variable>
  <xsl:variable name="xptrrangesuffix">
    <xsl:text>')))</xsl:text>
  </xsl:variable>

<xsl:output method="xml" indent="yes" encoding="ISO-8859-1" /> 

<xsl:template match="/">
 <xsl:apply-templates/>
</xsl:template>


<xsl:template match="@* | * | comment() | processing-instruction() | node()">
 <xsl:copy>
  <xsl:apply-templates select="@*|node()"/>
 </xsl:copy>
</xsl:template>

<!-- process out-of-file children (usually nite:child) -->
<xsl:template match="*[name(.)= $childel]">
  <xsl:variable name="href">
     <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select="@xlink:href"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="@href"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <xsl:variable name="sourcefilename" select="substring-before($href, '#')"/>
  <xsl:variable name="id1"> 
       <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select='substring-before(substring-after($href, $xptrlinkprefix), $xptrlinksuffix)'/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="substring-before(substring-after($href, '#id('), ')')"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <xsl:variable name="id2"> 
       <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select="substring-before(substring-after($href, $xptrrangeprefix),$xptrrangesuffix)"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="substring-before(substring-after($href, '..id('), ')')"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <!-- not sure why these two are necessary -->
  <xsl:variable name='startid' select='translate($id1,"&apos;","")'/>
  <xsl:variable name='endid' select='translate($id2,"&apos;","")'/>
  <xsl:variable name="file" select="document($sourcefilename,$docbase)"/>
  <xsl:choose>
    <xsl:when test="not($endid)">
      <!--
      <xsl:message>
        <xsl:text> Found an out-of-file child node, inserting content from file: </xsl:text>
        <xsl:value-of select="$sourcefilename"/>
        <xsl:text> at nite id: </xsl:text>
        <xsl:value-of select="$startid"/>
      </xsl:message>    
      -->
      <xsl:call-template name="insert-pseudochild">
        <xsl:with-param name="id" select="$startid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <!--
      <xsl:message>
        <xsl:text> Found nite child range, inserting content from: </xsl:text>
        <xsl:value-of select="$startid"/>
        <xsl:text> to: </xsl:text>
        <xsl:value-of select="$endid"/>
      </xsl:message>     
      -->
      <xsl:call-template name="insert-pseudochildren">
        <xsl:with-param name="startid" select="$startid"/>
        <xsl:with-param name="endid" select="$endid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
   </xsl:otherwise>
 </xsl:choose>
</xsl:template>

<!-- process out-of-file pointers (usually nite:pointer) -->
<xsl:template match="*[name(.) = $pointerel]">
  <xsl:variable name="role"><xsl:value-of select="@role"/></xsl:variable>
  <xsl:variable name="pname"><xsl:value-of select="name(..)"/></xsl:variable>
  <xsl:variable name="qualified-role" select="concat($pname, '-', $role)"/>
 <xsl:variable name="href">
     <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select="@xlink:href"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="@href"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <xsl:variable name="sourcefilename" select="substring-before($href, '#')"/>
  <xsl:variable name="id1"> 
       <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select='substring-before(substring-after($href, $xptrlinkprefix), $xptrlinksuffix)'/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="substring-before(substring-after($href, '#id('), ')')"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <xsl:variable name="id2"> 
       <xsl:choose>
        <xsl:when test="contains($linkstyle,'xpointer')">
           <xsl:value-of select="substring-before(substring-after($href, $xptrrangeprefix),$xptrrangesuffix)"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="substring-before(substring-after($href, '..id('), ')')"/>
        </xsl:otherwise>
      </xsl:choose>
  </xsl:variable>
  <!-- not sure why these two are necessary -->
  <xsl:variable name='startid' select='translate($id1,"&apos;","")'/>
  <xsl:variable name='endid' select='translate($id2,"&apos;","")'/>
  <xsl:variable name="file" select="document($sourcefilename,$docbase)"/>
  <xsl:element name="{$qualified-role}">
    <xsl:attribute name="ref"><xsl:value-of select="$href"/></xsl:attribute>
    <xsl:attribute name="pointer">true</xsl:attribute> 
    <xsl:choose>
      <xsl:when test="not($endid)">
        <!--
        <xsl:message>
          <xsl:text> Found a nite pointer node with role, </xsl:text> 
          <xsl:value-of select="$qualified-role"/>
          <xsl:text> inserting content from file: </xsl:text>
          <xsl:value-of select="$sourcefilename"/>
          <xsl:text> at nite id: </xsl:text>
          <xsl:value-of select="$startid"/>
        </xsl:message>
        -->
        <xsl:call-template name="insert-pointer">
          <xsl:with-param name="id" select="$startid"/>
          <xsl:with-param name="file" select="$file"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <!--
        <xsl:message>
          <xsl:text> Found a nite pointer node with role, </xsl:text> 
          <xsl:value-of select="$qualified-role"/>
          <xsl:text> inserting range content from file: </xsl:text>
          <xsl:value-of select="$sourcefilename"/>
          <xsl:text> from nite id: </xsl:text>
          <xsl:value-of select="$startid"/>
          <xsl:text> to nite id: </xsl:text>
          <xsl:value-of select="$endid"/>
        </xsl:message>
        -->
        <xsl:call-template name="insert-pointer-range">
          <xsl:with-param name="startid" select="$startid"/>
          <xsl:with-param name="endid" select="$endid"/>
          <xsl:with-param name="file" select="$file"/>
           
        </xsl:call-template>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:element>
</xsl:template>

<xsl:template name="insert-pseudochildren">
  <xsl:param name="startid"/>
  <xsl:param name="endid"/>
  <xsl:param name="file"/>
  <!--
  <xsl:message>
    <xsl:text>inserting</xsl:text>
    <xsl:value-of select="$startid"/>
    <xsl:text>to</xsl:text>
    <xsl:value-of select="$endid"/>
  </xsl:message>
  -->
  <xsl:choose>
    <xsl:when test="$startid = $endid">
      <xsl:call-template name="insert-pseudochild">
        <xsl:with-param name="id" select="$startid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <!-- insert the first id in the range -->
      <xsl:call-template name="insert-pseudochild">
        <xsl:with-param name="id" select="$startid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
      <!-- recurse on the range starting on the next id -->
      <xsl:call-template name="insert-pseudochildren">
        <!--      <xsl:with-param name="startid" select="$file//*[@nite:id=$startid]/following-sibling::*[position()=1]/@nite:id"/> -->
        <xsl:with-param name="startid" select="$file//*[attribute::*[name(.) = $idatt]=$startid]/following-sibling::*[position()=1]/attribute::*[name(.)=$idatt]"/>
        <xsl:with-param name="endid" select="$endid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template name="insert-pseudochild">
  <xsl:param name="id"/>
  <xsl:param name="file"/>
  <xsl:variable name="targetnode" select="$file//*[attribute::*[name(.)= $idatt] = $id]"/>
  <xsl:choose>
    <xsl:when test="$targetnode">
      <!--
      <xsl:message>
        <xsl:text>Inserting child for </xsl:text>
        <xsl:value-of select="$id"/>
      </xsl:message>
      -->
      <xsl:apply-templates select="$targetnode"/>   
    </xsl:when>
    <xsl:otherwise>
      <xsl:message>
        <xsl:text>Couldn't find id </xsl:text>
        <xsl:value-of select="$id"/>
        <xsl:text> for attribute </xsl:text>
        <xsl:value-of select="$idatt"/>
        <xsl:text> in file </xsl:text>
        <xsl:value-of select="name($file)"/>
        <xsl:text> as required to be an out-of-file child!</xsl:text>
      </xsl:message>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template name="insert-pointer">
  <xsl:param name="id"/>
  <xsl:param name="file"/>
  <xsl:variable name="targetnode" select="$file//*[attribute::*[name(.)= $idatt] = $id]"/>
  <xsl:choose>
    <xsl:when test="$targetnode">
      <xsl:variable name="elname" select="name($targetnode)"/>
      <xsl:element name="{$elname}">
        <xsl:for-each select="$targetnode/@*">
          <xsl:copy/>
        </xsl:for-each>
        <xsl:for-each select="$targetnode/text()">
          <xsl:copy/>
        </xsl:for-each>
      </xsl:element>
    </xsl:when>
    <xsl:otherwise>
      <xsl:message>
        <xsl:text>Couldn't find id </xsl:text>
        <xsl:value-of select="$id"/>
        <xsl:text> for attribute </xsl:text>
        <xsl:value-of select="$idatt"/>
        <xsl:text> in file </xsl:text>
        <xsl:value-of select="name($file)"/>
        <xsl:text> as required to be target of an out-of-file pointer!</xsl:text>
      </xsl:message>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template name="insert-pointer-range">
  <xsl:param name="startid"/>
  <xsl:param name="endid"/>
  <xsl:param name="file"/>
  <xsl:choose>
    <xsl:when test="$startid = $endid">
      <xsl:call-template name="insert-pointer">
        <xsl:with-param name="id" select="$startid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
    </xsl:when>
    <xsl:otherwise>
      <!-- insert the first id in the range -->
      <xsl:call-template name="insert-pointer">
        <xsl:with-param name="id" select="$startid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
      <!-- recurse on the range starting on the next id -->
      <xsl:call-template name="insert-pointer-range">
        <xsl:with-param name="startid" select="$file//*[attribute::*[name(.) = $idatt]=$startid]/following-sibling::*[position()=1]/attribute::*[name(.)=$idatt]"/>
        <xsl:with-param name="endid" select="$endid"/>
        <xsl:with-param name="file" select="$file"/>
      </xsl:call-template>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

</xsl:stylesheet>
