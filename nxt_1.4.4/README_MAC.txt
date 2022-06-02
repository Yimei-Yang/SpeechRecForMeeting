The NITE XML Toolkit - Working with media on Macs

The .command files that ship with NXT from 1.4.4 and those that come
with the separate corpus samples show the preferred way of working
with media files on the mac. If QuickTime is installed, media handling
will be passed to it via FMJ: http://fmj.sourceforge.net/.

We have come across some macs that require a slightly different
approach, so if you find the .command files fail to read media files
on your mac, please edit the command file using a text editor. You'll
find there's an alternative java call that can be un-commented (and
you should obviously comment out the current java call).

We apologise for this inconvenience and please get in touch via our
SourceForge page: https://sourceforge.net/projects/nite/ if you know
of a better solution or if neither java calls work on your mac.


 