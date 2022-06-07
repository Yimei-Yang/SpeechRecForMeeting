#!/bin/bash
# Generated by ant, January 5 2011
# Note that a Java runtime should be on the path.
# The current directory should be root of the nxt install.
# unless you edit this variable to contain the path to your install
# then you can run from anywhere.

# this magic incantation changes to the directory we're running from
cd "${0%/*}"

NXT="."
FMJHOME="$NXT/lib/fmj"

FMJJARS="$FMJHOME:$FMJHOME/fmj.jar:$FMJHOME/lib:$FMJHOME/lib/jdom.jar:$FMJHOME/lib/jffmpeg-1.1.0.jar:$FMJHOME/lib/jl1.0.jar:$FMJHOME/lib/jogg-0.0.7.jar:$FMJHOME/lib/jorbis-0.0.15.jar:$FMJHOME/lib/lti-civil.jar:$FMJHOME/lib/mp3spi1.9.4.jar:$FMJHOME/lib/tritonus_share.jar:$FMJHOME/lib/vorbisspi1.0.2.jar"

export CLASSPATH=".:$NXT:$NXT/lib:$NXT/lib/nxt.jar:$NXT/lib/pnuts.jar:$NXT/lib/resolver.jar:$NXT/lib/xalan.jar:$NXT/lib/xercesImpl.jar:$NXT/lib/xml-apis.jar:$NXT/lib/jmanual.jar:$NXT/lib/jh.jar:$NXT/lib/helpset.jar:$NXT/lib/poi.jar:$NXT/lib/eclipseicons.jar:$NXT/lib/icons.jar:$NXT/lib/forms.jar:$NXT/lib/looks.jar:$NXT/lib/necoderHelp.jar:$NXT/lib/videolabelerHelp.jar:$NXT/lib/dacoderHelp.jar:$NXT/lib/testcoderHelp.jar:$NXT/lib/prefuse.jar:$FMJJARS"

# the bootclasspath argument may seem redundant (and it can cause
# problems!) but on some macs, JMF is on the system CLASSPATH and this
# makes sure we override that. Comment / uncomment as necessary!
#java -Xms128m -Xmx512m -Xbootclasspath/p:$CLASSPATH -Djava.library.path="$FMJHOME" net.sourceforge.nite.nxt.GUI -corpus Data/meta/videolabeling-metadata.xml
java -Xms128m -Xmx512m -Djava.library.path="$FMJHOME" net.sourceforge.nite.nxt.GUI -corpus Data/meta/videolabeling-metadata.xml

		
