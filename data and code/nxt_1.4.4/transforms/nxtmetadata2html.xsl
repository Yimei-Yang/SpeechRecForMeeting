<?xml version="1.0" encoding="ISO-8859-1"?>
<xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="/">
<html>
    <head>
    	<title>NOM Metadata</title>
    	<meta name="description">
    		<xsl:attribute name="content"><xsl:value-of select="//corpus/@description"/>
    		</xsl:attribute>
    	</meta>
    	<meta name="generator" content="nxtmetadata2html.xsl" />
    	<meta name="keywords" content="nite nxt metadata" />
    	<meta http-equiv="content-type" content="text/html; charset=ISO-8859-1" />
        <link rel="stylesheet" type="text/css" href="metadata.css" /> 
    </head>
    <body>
    	<h1>NOM Metadata</h1>
        <h2>Reserved Attributes</h2>
        <p>These attributes apply to all relevant elements. (e.g. 
agentname does not apply to interaction codings and starttime and endtime do not apply to untimed elements.)</p>
        <xsl:apply-templates select="//reserved-attributes" />
    	<h2>Ontologies</h2>
    	<xsl:apply-templates select="//ontologies//ontology" />
		<hr />
    	<h2>Resources</h2>
    	<xsl:apply-templates select="//corpus-resources//code" />
    	<hr />
    	<h2>Agents</h2>
    	<hr />
    	<xsl:apply-templates select="//agents/*" />
    	<hr />
    	<h2>Codings</h2>
    	<hr />
    	<h3>Agent Codings</h3>
    	<xsl:apply-templates select="//codings/agent-codings//code" />
        <hr />
    	<h3>Interaction Codings</h3>
    	<xsl:apply-templates select="//codings/interaction-codings//code" />
    </body>
</html>
</xsl:template>

<xsl:template match="reserved-attributes">
  <!-- We don't know which names are defined, so we'll use a loop instead
of a template -->
  <dl>
    <xsl:for-each select="./*">
      <dt><b><xsl:value-of select="@name"/></b> is the attribute name for</dt>
      <dd><xsl:value-of select="name(.)"/></dd>
    </xsl:for-each>
  </dl>
</xsl:template>

<xsl:template match="ontology">
	<hr />
        <h4><a><xsl:attribute name="name">layer.<xsl:value-of select="@name" /></xsl:attribute><xsl:value-of select="@element-name" /></a></h4>
        <dl>
        <dt>Layer</dt><dd><xsl:value-of select="@name" /></dd>
        <!-- TODO: Insert entities from linked document here -->
        <dt>Filename</dt><dd><a><xsl:attribute name="href"><xsl:value-of select="@filename" />.xml</xsl:attribute><xsl:value-of select="@filename" />.xml</a></dd>
	<dt>Description</dt> <dd><xsl:value-of select="@description" /></dd> 
	<dt>Attributes</dt>
		<dd>
		<dl>
		<xsl:apply-templates select="attribute|pointer" />
		</dl>
		</dd>
	</dl>
</xsl:template>

<xsl:template match="agent">
	<h4>Agent name: <b><xsl:value-of select="@name" /></b></h4>
	<p>Description: <xsl:value-of select="@description" /></p>
</xsl:template>

<xsl:template match="code">
	<hr />
        <h4><a><xsl:attribute name="name">layer.<xsl:value-of select="parent::*/@name" /></xsl:attribute><xsl:value-of select="@name" /></a></h4> 
        <dl><dt>Layer</dt> 
        <dd><xsl:value-of select="parent::*/@name" /></dd>
	<dt class="attlist">Attributes</dt>
		<dd class="attlist">
		<dl>
		<xsl:apply-templates select="attribute|pointer" />
		</dl>
		</dd>
	</dl>
</xsl:template>

<xsl:template match="attribute">
	<dt class="attribute"><xsl:value-of select="@name" /></dt>
	<dd class="attribute"><xsl:value-of select="@value-type" /> 
    <xsl:for-each select="value"> : <i><xsl:value-of select="." /></i> :
    </xsl:for-each></dd>
</xsl:template>

<xsl:template match="pointer">
	<dt class="pointer"><xsl:value-of select="@role" /></dt>
        <dd class="pointer">Pointer to 
          <a><xsl:attribute name="href">#layer.<xsl:value-of select="@target" /></xsl:attribute><xsl:value-of select="@target" /></a></dd>
</xsl:template>

</xsl:stylesheet>

