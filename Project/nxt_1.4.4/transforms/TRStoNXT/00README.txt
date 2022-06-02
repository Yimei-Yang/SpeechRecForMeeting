
HOW TO CONVERT CHANNELTRANS TRANSCRIPTION DATA TO NXT
-----------------------------------------------------

The starting point for this process is assumed to be Channel Trans
output. ChannelTrans is a multi-channel version of Transcriber and
uses a similar file format (.trs). See 
 http://www.icsi.berkeley.edu/Speech/mr/channeltrans.html
for information on Channel Trans.

The scripts as they stand make some assumtions and have a set of
defaults. This README attempts to describe where assumptions are made
and what to change for your own situation.

The assumption is that text conforms to the Transcription Guidelines
for the AMI project available from this URL:
 http://corpus.amiproject.org/documentations/transcription
The end point is the 'reference' words and their parent segments for
import into NXT. 


Requirements
------------

A perl installation is assumed, including the perl modules XML::XPath
and XML::Parser which can be downloaded freely from cpan.org.

Java (at least 1.4.2) and an NXT installation are also required.



How to run
----------

To up-translate a single meeting from the channeltrans files to NXT,
without tokenizing & POS tagging the words, use this command:
  ./trs2nxt -o Out -i path/to/data/in/Transcriber-format -ob observationname -c your-metadata.xml

Further arguments to 'trs2nxt':
 -n <nxt_path> - the process assumes you have a valid NXT installation and you should
     pass it as the -n argument to trs2nxt. If you don't want to pass it every time 
     you can edit the script itself and set the NXTDIR variable.

 -we <name> - element name to use for 'words'. Defaults to 'w'.

 -wf <name> - filename to use for words. Defaults to 'words'. This is not a complete 
     filename but the 'name' part of the file name made up like this:
         <obs>.<agent>.<name>.xml

 -se <name> - element name to use for 'segments'. Defaults to 'segment'.

 -sf <name> - filename to use for segments. Defaults to 'segments'. This is not a 
     complete filename but the 'name' part of the file name made up like this:
         <obs>.<agent>.<name>.xml

This produces two files per agent: 
 words - the words as they appear in the transcription (see NOTES for further info)
 segments - the transcribers' segmentation of the file


Assumptions
-----------

Some defaults like the names of the elements used for words can be
overridden using command line arguments, but many assumptions are
coded into the scripts and can only be changed by direct edits to the
scripts. These are:

1. The mapping between Channel Trans channels and NXT agents is:
    channel 0 -> agent 'A'
    channel 1 -> agent 'B'
    channel 2 -> agent 'C'
    channel 3 -> agent 'D'
   to change this, edit trs2nxt and change the 'channels' and / or
   'agents' lists. These should have the same number of elements - the
   order implies the mapping.

2. The channel is stored as a cross reference on each segment element
  as attribute 'channel'. Turn off by editing 'split_words' and
  commenting out lines 259 and 260.

3. Punctuation content of words is noted and marked with a
 'punc="true"' attribute on the word element. Turn off by editing
 'split_words' and commenting out line 131.

4. IDs are assumed to be the default NXT attribute name:
 'nite:id'. Edit all instances in 'split_words' to change.

5. Start times are stored in attributes called 'starttime' and end
 times use attribute 'endtime'. Edit all instances in 'split_words' to 
 change.

6. Special characters are treated as in the AMI speech transcription
 guidelines (available at
  http://corpus.amiproject.org/documentations/transcription) The
 meaning of the symbols and their treatment can only be changed by
 editing the script: see the handleText routine of 'split_words'.

7. Segments and words go into separate files, i.e. they are assumed to
  be in different codings in the metadata file. This is a bit harder to
  change.
