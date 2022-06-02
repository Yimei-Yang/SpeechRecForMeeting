#!/usr/bin/perl

$outputdir="AvisynthScripts";
$framespersec=25;

while (<>) {
    if (/\t/ && /^(E|T|I)/) {
	($ob,$uid,$cam,$start,$end) = split(/\t/);
	$end=~s/\n//g;
	$uid=~s/\s//g;
	$cam=~s/\s//g;
	$framestart = int($start*$framespersec); # round down
	$frameend = int(($end*$framespersec)+0.5); # round up
	open(AVS, ">$outputdir/$ob\_$uid\_$start\_$end\.avs") || die "can't open $outputdir/$ob_$uid_$start_$end\.avs for write";
	print AVS "AudioDub(AVIsource(\"../Media/$ob\.$cam\_orig.avi\"), WAVSource(\"../Media/$ob.PreferredOverview.wav\"))\n";
	print AVS "Trim($framestart, $frameend)\n";
	close AVS;
    }
}

