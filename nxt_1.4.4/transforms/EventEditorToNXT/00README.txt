
HOW TO CONVERT EVENT EDITOR DATA TO NXT FORMAT
----------------------------------------------

This transform was devised for head and hand gesture coding on the AMI
corpus. However, it should be entirely generic. Any special cases
(e.g. handling of particular text content in a particular way will
either require manual changes to the Java program, or some
post-process. 


Requirements
------------

NXT installation with the appropriate CLASSPATH set up (include
nxt.jar; xalan.jar on your CLASSPATH at least). First compile
EventEditorToNXT.java.


Running
-------

To transform one file use
 java EventEditorToNXT -i <input_file> -t <tagname> -a <attname> -s <starttime> -e <endtime> -c <comment> [ -l <endtime> ]

The arguments are the names of the elements and attributes to be
output. Because EventEditor is event based, the last event does not
have an end time. If you want an end time to appear in the NXT format,
use the '-l' argument.

IDs are not added to elements, but you can use the provided
'add-ids.xsl' stylesheet for that:

 java -classpath $NXTDIR/lib/xalan.jar org.apache.xalan.xslt.Process -in <outputfromabove> -out <outfile> -xsl add-ids.xsl -param session <observationname> -param participant <agentname>

where NXTDIR is your local NXT install, or you can poimnt to anywhere
you happen to have installed Xalan. At least the session parameter,
and really the participant one too, should be used as these help the
IDs to be unique.

You may well want to batch this process, but that will normally be a
corpus-specific operation and is not covered here.

Jonathan Kilgour - last edit 27.02.07


