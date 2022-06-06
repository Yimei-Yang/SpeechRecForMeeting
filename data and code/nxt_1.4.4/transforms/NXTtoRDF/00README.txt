
CONVERTING NXT DATA TO RDF FORMAT
---------------------------------

This transform comprises an NXT application that produces RDF in the
Turtle format (see http://www.dajobe.org/2004/01/turtle/)

Running is fairly simple - for example use:

 java NXTtoRDF -c /disk/scratch/vvv/Data/AMI/NXT-format/AMI-metadata.xml -n ami -oc -o ES2002a > ES2002a.turtle

to transform a single observation (ES2002a) from a corpus called 'ami'
using explicitly ordered children. The -oc argument makes a separate
Turtle node for every parent child relation in the corpus and
explicitly states the order of children in each. This makes the
transform more complete but adds to the number of RDF triples and
output file size considerably.

Requirements
------------

NXT installation with the appropriate CLASSPATH set up (include
nxt.jar; xalan.jar on your CLASSPATH at least). First compile
NXTtoRDF.java.



Issues
------

Output goes onto standard out; other messages appear on stderr. Output
can be really huge!

This process turns lazy loading off, so you may find you run out of
memory for very large observations / corpora. You can increase Java
heap space using -Xms -Xmx arguments (some sites state that setting
these to equal values is the most efficient thing to do). Try e.g.
 java -Xmx512m ...


Jonathan Kilgour - last edit 20.07.07

