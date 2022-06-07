# observer2nxt.perl                                  JEAN CARLETTA
# FEB 04  
# 
# This is the best model to date for a general Observer translation, but
# it relies on lookup tables detailing the codes and what they
# should look like in NXT for the translation, and for separating
# the Observer codes into NXT data streams by type.  If we read
# from the Observer configuration and had some convention for
# XML tag naming, we wouldn't need the tables.
#
# The odf example I'm working from has funny end of lines (CRLF, I think).
# I'm solving this by chopping twice.  This may not be a robust approach.
## late change - problem with accidentally writing modifiers on nulltags.
## plus last tag being written without modB.
#-------------------------------------------------------------------
if ($#ARGV < 0) {
  die "Usage:   perl observer2nxt.perl OBS-NAME";
}


$obs = shift(@ARGV);
open(IN,"<$obs.odf") or die "Can't open $obs.odf";

# we have five types of codes, and two agents.  We have to classify codes
# by type and agent because this determines what coding file they
# end up in.  Types of codes are behavioural classes in Observer
# terms and agents are subjects; both can be read off the configuration.

@types = ("LH-gesture","RH-gesture");
@agents= ("A","B");

# look up table, @type, for what codes go with what behavioural classes (or,
# in xml terms, what tags belong in what codings).  This could be
# retrieved from the configuration file rather than hand-authored.
fill_type_lookup();

# look up table, @code, for what we read as an Observer code to what we write
# as an xml tag.  Observer codes can have spaces in them. 
##In general, a translation can be done by string manipulation into something suitable.
## removes blanks and ampersands at least.
fill_code_lookup();

# look up table, @modcode, for what we read as Observer modifers.
# to what we write.  Same comments on general manipulation.   
## Any code can be configured to take up to two modifiers, but a code is
## restricted to what modifiers it can take.  We ignore this and just throw
## all the modifiers together into one big lookup, and look for them with
## every code. I don't think this breaks but it doesn't validate.
fill_modcode_lookup();


# We need a filehandle for each xml output file (one per agent times type).
# This fills the look up table, @stream, with associations between whatever
# we get by calling codingindex on an agent and type and the filehandle.
## I've made this so general because I've had trouble with filehandles before
## but it's probably more complex than it needs to be.
fill_stream_lookup();

# Most Observer behavioural classes give states, not events.  That means there
# is an initialize code at time 0.00 which is considered to run until the
# first "proper" code, and all codes run from when they are entered until
# the next code in the same class for the same agent.  That means we have to
# save information about the last code encountered for each coding.
## We don't consider the case of event codings, which I believe are possible
## in The Observer.
fill_last_lookup();

# We need unique ids for tags; it's useful (for human readability) if they're
# sequential.  Make some counters, one per coding file, in a lookup on
# codingindexes, $id.
make_id_counters();

# For each type, there is probably a tag which really means "no such behaviour
# is happening".  To make a decent translation, we want tags with the names
# e.g. gesture and no-gesture, with a type attribute on just the gestures.
# Make a lookup table, @nulltags, for the null cases that takes us from types
# to how observer represents the null code.
##I think this behaviour shouldn't go in the general translator because I don't
## think these cases are marked in the configuration; they're just common practice.
fill_nulltags();

# open the output files and write a root tag in each one.
# every time we print we need to select the correct filehandle first.
# naming it in the print statement doesn't work (it gets treated
# as a string rather than a filehandle).  That's why we have a special
# routine for this.


$i = 0; 
while ($i < scalar(@types)) {
  $j = 0;
  while ($j < scalar(@agents)) {
     $maxi = scalar(@types);
     $maxj = scalar(@agents);
     #selectprint(STDOUT,"doing $agents[$j], $types[$i] I is $i J is $j maxi is $maxi maxj is $maxj\n");
     #selectprint(STDOUT,"codingindex is" . codingindex($agents[$j],$types[$i])."\n");
     #selectprint(STDOUT,"stream is" . $stream{codingindex($agents[$j],$types[$i])}."\n");
     open($stream{codingindex($agents[$j],$types[$i])},">$obs.$agents[$j].$types[$i].xml");
     selectprint($stream{codingindex($agents[$j],$types[$i])},"<nite:root xmlns:nite=\"http://nite.sourceforge.net/\" id=\'$obs.$agents[$j].$types[$i].root\'>\n");
     $j = $j+ 1;
  }
  $i = $i + 1;
};


#read the odf file one line at a time, looking for lines that
#start with timestamps.  Each time we get a tag, that's the start
#time for it.  Save what the last tag was of each type in lasttag,
#with the starting timestamp in last.  When we get to the next
#tag of the same type, we can write the previous one, using
#the tag from lasttag, the starttime from last, and the end
#time from the current line.  Last is initialized with -1, allowing
#us to test whether we've seen any previous tags before.

# Example full line:
# timestamp agent,tag,modA,modB, /* comment
# agent is A or B
# tag (contains spaces) comes from one of N categories:
# LH gesture: 	(example: LHPoint)
# RH gesture: (example: RHPoint)
#
# the LH and RH gesture tags are the same but we don't bother parsing
# them that way because The Observer doesn't consider them that way (they're
# separate behavioural classes) and NXT can't either (e.g., can't multiply
# agent by type by hand).

# 3.90 A,RH ref point,G,stationary, /* here
while($line = <IN>) {
  chop($line);
  chop($line); # this may not be needed on windows; cr/lf problem
  #selectprint(STDOUT,"$line\n");
  # The last tag is always a control code, {end}.
  # If we're there, write all types of outstanding codes using this timestamp
  # some files got the final bracket of the code truncated
  # so search on {end, not {end}
  $timestamp="";
  $agent="";
  $modA="";
  $modB="";
  $comment="";
  if ($line =~ /^([0-9]+.[0-9][0-9])\s{end/) {
      $timestamp = $1;
      $i = 0; 
      while ($i < scalar(@types)) {
        $j = 0;
         while ($j < scalar(@agents)) {
             writetag($agents[$j], $types[$i], $lasttag{codingindex($agents[$j],$types[$i])},$last{codingindex($agents[$j],$types[$i])}, $timestamp,  $lastmodA{codingindex($agents[$j],$type{$tag})}, $lastmodB{codingindex($agents[$j],$type{$tag})}, $lastcomment{codingindex($agent[$j],$type{$tag})});
            $j = $j+ 1;
         }
         $i = $i + 1;
      }
    }
  elsif ($line =~ /^([0-9]+.[0-9][0-9])\s([AB])/) {
     $timestamp = $1;
     $agent = $2;
     $rest = $';
     if ($rest =~ /,/) { # then we've got some more
         if ($rest =~ /\/\*/) { #then we have comment
            $comment = $';
            $rest = $`;
          };
         # pick up the bits one by one
         if ($rest =~ /^,([^,]+)/) { #get the tag
            $tag = $1;
            $rest = $';
         };

         if ($rest =~ /^,([^,]+)/) { #then we have first subtag      
            $modA = $1;
            $rest = $';
         };
   
         if ($rest =~ /^,([^,]+)/) { #then we have second subtag      
            $modB = $1;
            $rest = $';
         };
     # else there's nothing on the line but the tag
     } else {
        $tag = $rest;
     } ;
     # now find out if the tag is handed
     #if ($tag =~ /^([RL])H /) {
     #    $hand = $1;
     #    $tag = $';
     #};

     if ($last{codingindex($agent,$type{$tag})} >= 0) {

        if ($type{$tag}) {
           writetag($agent, $type{$tag}, $lasttag{codingindex($agent,$type{$tag})},$last{codingindex($agent,$type{$tag})}, $timestamp, $lastmodA{codingindex($agent,$type{$tag})}, $lastmodB{codingindex($agent,$type{$tag})},  $lastcomment{codingindex($agent,$type{$tag})});
        }
	else {
          selectprint(STDOUT, "Found something I can't handle: $tag!\n");
	};
      };
      #store the timestamp and tag you found this time to use as start time
      #and tag when encounter next tag of this type
      $last{codingindex($agent,$type{$tag})} = $timestamp;
      $lasttag{codingindex($agent,$type{$tag})} = $code{$tag};
      $lastmodA{codingindex($agent,$type{$tag})} = $modcode{$modA};
      $lastmodB{codingindex($agent,$type{$tag})} = $modcode{$modB};
      # for the comment, just make sure there's no ampersands or angle brackets to trip on
      $lastcomment{codingindex($agent,$type{$tag})} = smooth($comment);

   };
};

## THERE SHOULD BE CODE HERE TO HANDLE NOTES.  These come after the {notes} line,
## which is after the timestamped {end} code.  Notes *should* have a time given
## in hh:mm:ss:dd format, whitespace, and then the note.
## My input examples look hand-munged so I'm putting this off for now.

close(IN);

$i = 0; 
while ($i < scalar(@types)) {
  $j = 0;
  while ($j < scalar(@agents)) {
    selectprint($stream{codingindex($agents[$j],$types[$i])},"</nite:root>\n");
    close($stream{codingindex($agents[$j],$types[$i])});
    $j = $j+ 1;
  }
  $i = $i + 1;
}



sub fill_stream_lookup{
$i = 0; 
while ($i < scalar(@types)) {
  $j = 0;
  while ($j < scalar(@agents)) {
     $stream{codingindex($agents[$j],$types[$i])} = $agents[$j].$types[$i]; # gensym didn't work?
     $j = $j+ 1;
  }
  $i = $i + 1;
}
}


sub selectprint{
  my($fh, $string) = @_;
  select($fh);
  print $string;
}

sub selectprintdur{
  my($fh, $string) = @_;
  select($fh);
  printf(qq[%.2f],$string);
}

sub make_id_counters{
$i = 0; 

while ($i < scalar(@types)) {
  $j = 0;
  while ($j < scalar(@agents)) {
     $id{codingindex($agents[$j],$types[$i])} = 0;
     $j = $j+ 1;
  }
  $i = $i + 1;
}
}

sub new_id{
  my($codetype) = @_;
  $id{$codetype} = $id{$codetype} + 1;
  return $id{$codetype};
}

sub fill_last_lookup{
$i = 0; 

while ($i < scalar(@types)) {
  $j = 0;
  while ($j < scalar(@agents)) {
       $last{codingindex($agents[$j],$types[$i])} = -1;
     $j = $j+ 1;
  }
  $i = $i + 1;
}
}
# change the strings here if you don't like the tag names,
# and change the metadata xml file to match.
## if we had nulltags entries match Observer originals
## instead of the same translations as these, we wouldn't
## need translations for the nulltags.  
sub fill_code_lookup{
   $code{'LHPoint'} = 'LH-point';
   $code{'LH no gesture'} = 'LH-no-gesture';

   $code{'RHPoint'} = 'RH-point';
   $code{'RH no gesture'} = 'RH-no-gesture';
}

sub fill_nulltags{
   $nulltags{'LH-gesture'} = 'LH-no-gesture';
   $nulltags{'RH-gesture'} = 'RH-no-gesture';
}


sub fill_modcode_lookup{
   $modcode{'Missing gesture type'} = 'uncodedshape';
   $modcode{'G'} = 'G';
   $modcode{'pinkie'} = 'pinkie';
   $modcode{'whole hand'} = 'hand';
   $modcode{'pen'} = 'pen';
   $modcode{'2 finger'} = 'twofinger';
   $modcode{'iconic'} = 'iconic';
   $modcode{'beat'} = 'beat';
   $modcode{'other'} = 'other';
   $modcode{'Missing gesture type'} = 'uncodedmovement';
   $modcode{'trace'} = 'trace';
   $modcode{'up & down'} = 'up-n-down';
   $modcode{'squiggly line'} = 'squiggly';
   $modcode{'circular ref'} = 'circular';
   $modcode{'staccato'} = 'staccato';
   $modcode{'stationary'} = 'stationary';
   $modcode{'tap'} = 'tap';
   $modcode{'other'} = 'other';
   $modcode{'place marker'} = 'place marker';

}

sub smooth{
  my($string) = @_;
  $string =~ s/<//;
  $string =~ s/>//;
  $string =~ s/&/and/;
  $string =~ s/"/'/;
  return $string;
}


sub fill_type_lookup{
   $type{'LHPoint'} = 'LH-gesture';
   $type{'LH no gesture'} = 'LH-gesture';
   $type{'RHPoint'} = 'RH-gesture';
   $type{'RH no gesture'} = 'RH-gesture';

}

sub code_has_type{
  my($tag, $tagtype) = @_;
  if ($type{$tag} == $tagtype) {
     return 1;
   }
   else {
     return 0;
   }
}

## we shouldn't write modifiers and comments if they're empty
sub writetag{
  my($agent, $type,$tag, $start, $end, $modA, $modB, $comment) = @_;
  $id = new_id(codingindex($agent,$type));
  $dur = $end-$start;
  # if it's the null tag for this category, then write that as the tag name;
  # otherwise, write the type name as the element name and a type attribute with the tag.
  if ($tag eq $nulltags{$type}) {
     selectprint($stream{codingindex($agent,$type)},"   <$tag id=\'$obs.$agent.$type.$id\' start=\'$start\' end=\'$end\'  duration=\'");
     #hackery to limit duration to two decimal places.
     selectprintdur($stream{codingindex($agent,$type)}, $dur);
     selectprint($stream{codingindex($agent,$type)}, "\'");
   } else {
      selectprint($stream{codingindex($agent,$type)},"   <$type type=\'$tag\' id=\'$obs.$agent.$type.$id\' start=\'$start\' end=\'$end\'  duration=\'");
     #hackery to limit duration to two decimal places.
     selectprintdur($stream{codingindex($agent,$type)}, $dur);
     selectprint($stream{codingindex($agent,$type)}, "\'");
     if ($modA) {
        selectprint($stream{codingindex($agent,$type)}," modifierA=\'$modA\'");
     };
     if ($modB) {
        selectprint($stream{codingindex($agent,$type)}," modifierB=\'$modB\'");
     };
  }
  ## use double quotes on comment because we want to allow apostrophes inside
  if ($comment) {
     selectprint($stream{codingindex($agent,$type)}," comment=\"$comment\"");
  };
 selectprint($stream{codingindex($agent,$type)},"\/>\n");
}

sub codingindex{
  my($agent,$type) = @_;
  $index = $agent.$type;
  return $index;
}
 
