Converting NXT timed data to Media files
----------------------------------------

Sometimes one wishes to create video or audio snippets corresponding
to timed occurrences in an NXT corpus. This directory describes some
approaches to doing that.


Prerequisites:
-------------
 perl
 AviSynth
 VirtualDub

You may well be able to use a different video / audio editing process,
but we have experienced problems with some that don't chop video at
exact times, instead waiting for the nearest keyframe. It's also
sometimes difficult to add an audio track in some tools. The best
approach we have found so far is this (unfortunately) PC-only
approach. avidemux may provide a cross-platform alternative and
scripts for such a transform would fit into this directory.


Steps:
-----

1. Create a tab-delimited file from your NXT data 
  Using FunctionQuery it's pretty easy to extract the data you want
  from NXT. The format expected by the next step is a file where each
  line contains these tab-delimited fields: Observation_name,
  Participant_ID, Media_ID, Start, End. For example: ES2002a FEE005
  Closeup4 113.21 116.66 An example of how to produce such a file for
  the AMI corpus is included below.

2. Transform to a set of AviSynth files
   mkdir AvisynthScripts
   perl transform.pl results.txt
  Saves results in a directory called AvisynthScripts (can be a large
  number of files). The script makes certain assumtions about the
  naming and location of media files that may be valid for the AMI
  corpus but will need to be changed for your own purpose. There has
  been no attempt to parameterize any part of the script so if your
  videos have a different frame rate you have to change that too.

3. Load the files in turn into VirtualDub. Note that this process can
  be batched easily using VirtualDub but that you should first make
  sure VirtualDub has a sensible compression set otherwise output
  files can be very big. Files will be saved with an .avs extension,
  but that can be changed using a shell script or similar.

  In order to batch process the files in VirtualDub, first set and
  check your video compression - DON'T leave it on the default setting
  which is uncompressed - recommend XviD which is a free version of
  DivX. Then start batch wizard and drag and drop your avs files in
  the window (note that you can apply a filter and I normally convert
  .avs to .avi). It's also worth routing output to a different
  folder. Hit "Add to Queue" and select "Save as .AVI". The filenames
  disappear from your list. They don't start executing though until
  you bring up the Job Control list, select the first undone job and
  click "Start".


FunctionQuery Example for the AMI corpus
----------------------------------------

java FunctionQuery -c AMI-metadata.xml -q '($s subj)($st subj-type):$s>$st
&& $st@name="positive-subjective"' -atts obs '@extract(($sp speaker)($m
meeting):$m@observation=$s@obs && $m^$sp & $s@who==$sp@nxt_agent,
global_name, 0)' '@extract(($sp speaker)($m meeting):$m@observation=$s@obs
&& $m^$sp & $s@who==$sp@nxt_agent, camera, 0)' starttime endtime > results

This will create a single file containing results for the entire AMI
corpus.



Jonathan Kilgour
27/05/2010
