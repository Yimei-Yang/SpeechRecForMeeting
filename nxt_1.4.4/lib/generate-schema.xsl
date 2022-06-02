<?xml version="1.0"?>

<!-- Jonathan Kilgour 31/1/3
  a stylesheet for generating a schema file from a NITE metadata
  format file. 

  Altered 21/3/3 to cope with simple as well as standoff corpora
-->

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
   version="1.0" xmlns:nite="http://nite.sourceforge.net/"
	xmlns:xsd="http://www.w3.org/2001/XMLSchema">

 <xsl:template match="/">
  <xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema"  
            xmlns:nite="http://nite.sourceforge.net/"><xsl:text>
  </xsl:text>
  <xsd:import namespace="http://nite.sourceforge.net/" schemaLocation="typelib.xsd"/><xsl:text>
  </xsl:text>
<xsd:element name="stream" type="nite:Stream">
  <xsd:unique name="stream-uniqueness-constraint">
   <xsd:selector xpath="*"/>
   <xsd:field xpath="@nite:id"/>
  </xsd:unique>
</xsd:element><xsl:text>
  </xsl:text>
    <xsl:choose>
      <xsl:when test="corpus/@type='simple'">
       <xsl:apply-templates select="corpus/codings"/>
      </xsl:when>
      <xsl:otherwise>
       <xsl:apply-templates select="corpus/codings/*/coding-file"/>
       <xsl:apply-templates select="corpus/ontologies"/>
       <xsl:apply-templates select="corpus/object-sets"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsd:schema>
 </xsl:template>


 <xsl:template match="object-sets">
     <xsl:apply-templates select="object-set"/>
 </xsl:template>

<!-- can't quite remember the rules for object sets -->

 <xsl:template match="ontologies">
     <xsl:apply-templates select="ontology"/>
 </xsl:template>

 <xsl:template match="ontology">
   <xsl:element name="xsd:element">
     <xsl:attribute name="name"><xsl:value-of select="@element-name"/></xsl:attribute>
  <xsd:complexType>
   <xsd:complexContent>
    <xsd:extension base="nite:TypeDefinition">
     <xsd:sequence minOccurs="0" maxOccurs="unbounded">
      <xsl:element name="xsd:element">
       <xsl:attribute name="ref"><xsl:value-of select="@element-name"/></xsl:attribute>
      </xsl:element>
     </xsd:sequence>
     <xsl:apply-templates select="attribute" mode="print-attributes"/>
    </xsd:extension>
   </xsd:complexContent>
  </xsd:complexType>   
   </xsl:element>
 </xsl:template>

 <xsl:template match="coding-file">
     <xsl:apply-templates/>
 </xsl:template>

 <xsl:template match="structural-layer">
     <xsl:apply-templates mode="structural"/>
 </xsl:template>

 <xsl:template match="featural-layer">
     <xsl:apply-templates mode="featural"/>
 </xsl:template>

 <xsl:template match="time-aligned-layer">
     <xsl:apply-templates mode="structural"/>
 </xsl:template>

<!-- Note that if the "text-content" attribute of a code is true, no
content is permitted either diretly or via pointers or children. -->
 <xsl:template match="code" mode="structural">
   <xsl:element name="xsd:element">
     <xsl:attribute name="name"><xsl:value-of select="@name"/></xsl:attribute>
     <xsd:complexType> 
      <xsl:choose>
      <xsl:when test="@text-content='true'">
       <xsd:simpleContent>
        <xsd:extension base="xsd:string">	
         <xsl:apply-templates select="attribute" mode="print-attributes"/>
         <xsd:attributeGroup ref="nite:nite-timed-attributes"/>
        </xsd:extension>	
       </xsd:simpleContent>
      </xsl:when>
      <xsl:otherwise>
       <xsd:complexContent>
        <xsd:extension base="nite:StructuralToken">
         <xsd:sequence minOccurs="0" maxOccurs="unbounded">
          <xsd:choice>
           <xsl:apply-templates select=".." mode="print-pointed-to-codes"/>
           <xsl:apply-templates select="pointer" mode="print-pointer-code"/>
          </xsd:choice>
         </xsd:sequence>
        <xsl:apply-templates select="attribute" mode="print-attributes"/>
        </xsd:extension>
       </xsd:complexContent>
      </xsl:otherwise>
     </xsl:choose>
    </xsd:complexType>
   </xsl:element>

   <xsl:apply-templates select="pointer" mode="print-pointer-element"/>
 </xsl:template>


 <xsl:template match="code" mode="featural">
   <xsl:element name="xsd:element">
     <xsl:attribute name="name"><xsl:value-of select="@name"/></xsl:attribute>
     <xsd:complexType> 
      <xsl:choose>

      <xsl:when test="@text-content='true'">
       <xsd:simpleContent>
        <xsd:extension base="xsd:string">	
         <xsl:apply-templates select="attribute" mode="print-attributes"/>
         <xsd:attributeGroup ref="nite:nite-attributes"/>
        </xsd:extension>	
       </xsd:simpleContent>
      </xsl:when>

      <xsl:otherwise>
       <xsd:complexContent>
        <xsd:extension base="nite:FeaturalToken">
         <xsd:sequence minOccurs="0" maxOccurs="unbounded">
          <xsd:choice>
           <xsl:apply-templates select=".." mode="print-pointed-to-codes"/>
           <xsl:apply-templates select="pointer" mode="print-pointer-code"/>
          </xsd:choice>
         </xsd:sequence>
        <xsl:apply-templates select="attribute" mode="print-attributes"/>
        </xsd:extension>
       </xsd:complexContent>
      </xsl:otherwise>

     </xsl:choose>

    </xsd:complexType>
   </xsl:element>

   <xsl:apply-templates select="pointer" mode="print-pointer-element"/>

 </xsl:template>



 <xsl:template match="structural-layer|featural-layer|time-aligned-layer" mode="print-pointed-to-codes">
   <xsl:if test="@points-to">
    <xsl:variable name="pointsto"><xsl:value-of select="@points-to"/></xsl:variable>
    <xsl:apply-templates select="//*[@name=$pointsto]" mode="print-codes"/>
   </xsl:if>
   <xsl:if test="@recursive-points-to">
    <xsl:variable name="rpointsto"><xsl:value-of select="@recursive-points-to"/></xsl:variable>
    <xsl:apply-templates select="." mode="print-codes"/>
    <xsl:apply-templates select="//*[@name=$rpointsto]" mode="print-codes"/>
    <xsd:element ref="nite:child"/>                 
   </xsl:if>
   <xsl:if test="@recursive='true'">
    <xsl:apply-templates select="." mode="print-codes"/>
<!--	    <xsd:element ref="nite:child"/>                 -->
   </xsl:if>
 </xsl:template>


 <xsl:template match="ontology" mode="print-codes">
   <xsl:element name="xsd:element">
     <xsl:attribute name="ref"><xsl:value-of select="@element-name"/></xsl:attribute>
   </xsl:element>
 </xsl:template>

 <xsl:template match="structural-layer|featural-layer|time-aligned-layer" mode="print-codes">
   <xsl:apply-templates mode="print-codes"/>
 </xsl:template>

 <xsl:template match="code" mode="print-codes">
   <xsl:element name="xsd:element">
     <xsl:attribute name="ref"><xsl:value-of select="@name"/></xsl:attribute>
   </xsl:element>
 </xsl:template>


<!-- ATTRIBUTES -->
<!-- these can be one of three types: 
      string - xsd:string
      number - xsd:decimal
      enumeration - restriction of xsd:string with xsd:enumeration 
-->
 <xsl:template match="attribute" mode="print-attributes">
   <xsl:element name="xsd:attribute">
     <xsl:attribute name="name"><xsl:value-of select="@name"/></xsl:attribute>
     <xsl:if test="@value-type='string'">
      <xsl:attribute name="type">xsd:string</xsl:attribute>
     </xsl:if>
     <xsl:if test="@value-type='number'">
      <xsl:attribute name="type">xsd:decimal</xsl:attribute>
     </xsl:if>
     <xsl:if test="@value-type='enumerated'">
      <xsl:element name="xsd:simpleType">
       <xsd:restriction base='xsd:string'>
	<xsl:apply-templates select="value" mode="define-enumerations"/>
       </xsd:restriction>
      </xsl:element>
     </xsl:if>
   </xsl:element>
 </xsl:template>

 <xsl:template match="value" mode="define-enumerations">
    <xsl:element name="xsd:enumeration">
      <xsl:attribute name="value"><xsl:apply-templates/></xsl:attribute>
    </xsl:element>
 </xsl:template>

 <xsl:template match="pointer" mode="print-pointer-code">
   <xsl:element name="xsd:element">
     <xsl:attribute name="ref"><xsl:value-of select="concat(../@name, '-', @role)"/></xsl:attribute>
   </xsl:element>
 </xsl:template>

 <xsl:template match="pointer" mode="print-pointer-element">
   <xsl:element name="xsd:element">
     <xsl:attribute name="name"><xsl:value-of select="concat(../@name, '-', @role)"/></xsl:attribute>
  <xsd:complexType>
   <xsd:complexContent>
    <xsd:extension base="nite:Role">
    <xsd:sequence maxOccurs="1" minOccurs="0">                
    <xsd:choice>                
    <xsl:variable name="target"><xsl:value-of select="@target"/></xsl:variable>
    <xsl:apply-templates select="//*[@name=$target]" mode="print-codes"/>
    </xsd:choice>                
    </xsd:sequence>
    </xsd:extension>
   </xsd:complexContent>
  </xsd:complexType>
   </xsl:element>
 </xsl:template>

</xsl:stylesheet>
