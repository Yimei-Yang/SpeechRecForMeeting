import java.util.*;
import java.io.File;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.*;
import javax.swing.*;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.gui.textviewer.*;
import net.sourceforge.nite.gui.util.*;
import net.sourceforge.nite.gui.mediaviewer.*;
import net.sourceforge.nite.nxt.*;
import net.sourceforge.nite.nstyle.handler.*;
import net.sourceforge.nite.time.*;

import net.sourceforge.nite.nom.NOMException;

/**
 * Changing Dagmar sample to work with monitor data.
 * 
 * Problems:
 * 
 * The return order of values for ($m move)($a word):($m@id=ID) and ($m ^ $a)
 * is all words at the first level and then all at the second (some are within
 * intervening references).   
 * 
 * The coding goes by so fast that it's difficult to understand it.
 * So this perhaps isn't the best demo example.
 * 
 * We set up time synchronization for the timing elements (start,
 * thirty-second-warning, and finish) but they are instantaneous, 
 * with the same start and end time- and somehow the interface 
 * doesn't know to un-highlight them when the time has gone by.
 * Same start and end is out of spec, I think, so I've just removed
 * this.
 * 
 * I wonder if something about the highlighting changes have messed
 * up the ability to play the signal associated with a timed element -
 * although I think I get it using crtl-right-click, as long as I've
 * already left-clicked to get some kind of blue highlight on the 
 * correct element.  I think maybe the timings are so short sometimes
 * that they can feel out of synch with what is actually played -
 * either that or the wrong thing is playing.  But this is probably
 * a knock-on effect from the original coding, which won't have been
 * that accurate in timing terms.
 * 
 * @author Jonathan Kilgour, Jean Carletta March 2003
 **/

public class MonitorDisplay {
    private Clock niteclock;
    private NITEVideoPlayer video;
    NOMWriteCorpus nom;
    NiteMetaData controlData;
    JInternalFrame iframe;
    JInternalFrame iframe2;
    String corpusname;
    String observationname;
    JFrame frame;
    String exportdir = ".";
    //    NTextArea gestureTextArea;
    NTextArea feedbackgazeTextArea;
    NTextArea feedbackTextArea;
    NTextArea routegazeTextArea;
    //NTextArea timingTextArea;
    NTextArea transcriptionTextArea;
    NTree ntree;
    JScrollPane pane;
    JDesktopPane desktop;
    NOMWriteElement changed;
    private net.sourceforge.nite.search.GUI search=null;
    
    private Engine searchEngine = new Engine();
    
    public MonitorDisplay(String c, String o) {
	corpusname = c;
	observationname = o;
	
	try {
	    controlData = new NiteMetaData(corpusname);
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}
	
	if (observationname==null) {
	    ChooseObservation co = new ChooseObservation(controlData);
	    observationname=co.popupDialog();
	} 
	
	niteclock = new DefaultClock();
	if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
	    System.out.println("This is a standoff corpus: NOM being loaded");
	    try {
		nom = new NOMWriteCorpus(controlData);
		NiteObservation obs = observationWithName(observationname);
		nom.loadData(obs);
		System.out.println("Finished loading.");
		search=new net.sourceforge.nite.search.GUI(nom);
		setupInterface(nom);
	    } catch (NOMException nex) {
		nex.printStackTrace();
	    }
	} else {
	    System.out.println("This is a standalone or simple corpus: no NOM has been loaded");
	}
    }
    
    private void setupInterface(NOMWriteCorpus nom) {
	frame = new JFrame();
	JMenuBar menubar = new JMenuBar();
	JMenu file = new JMenu("File");
	//file.add(new SaveAction());
	file.add(new ExitAction());
	menubar.add(file);
	/* Make the search menu too! */
	JMenu search = new JMenu("Search");
	search.add(new SearchAction());
	menubar.add(search);
	frame.setJMenuBar(menubar);
	desktop = new JDesktopPane();
	
	//	JTextArea jta = (JTextArea) ntah.displayElement();
	//	jta.setLineWrap(true);
	
	addTextAreaFeedbackGazeContent();
	addTextAreaFeedbackContent();
	addTextAreaRouteGazeContent();
	//addTextAreaTimingContent();
	addTextAreaTranscriptionContent();
	NSignal sig = (NSignal) controlData.getSignals().get(0);
	// first signal;
	if (sig.getMediaType() == NSignal.VIDEO_SIGNAL) {
	    String video_filename =
		controlData.getSignalPath()
		+ File.separator
		+ observationname
		+ "."
		+ sig.getName()
		+ "."
		+ sig.getExtension();
	    System.out.println("Video file: " + video_filename);
	    video =
		new NITEVideoPlayer(
				    new File(video_filename),
				    (DefaultClock) niteclock);
	    video.setLocation(new Point(440, 0));
	    video.setSize(new Dimension(450, 400));
	    video.setID(3);
	    
	    // niteclock.registerTimeHandler((TimeHandler)video);
	    //	    video.setVisible();
	    desktop.add(video);
	}
	
	JPanel panel = new JPanel();
	desktop.setSize(new Dimension(900, 650));
	frame.getContentPane().add(desktop);
	frame.setSize(new Dimension(900, 650));
	frame.setVisible(true);
    }

    private void clearTextArea(TextAreaHandler nta) {

    }
    
    private void addTextAreaFeedbackGazeContent() {
	iframe =
	    new JInternalFrame("FeedbackGaze Display", true, false, true, true);
	iframe.setSize(new Dimension(220, 320));
	iframe.setLocation(new Point(0, 0));
	if (feedbackgazeTextArea != null) {
	    niteclock.deregisterTimeHandler((TimeHandler) feedbackgazeTextArea);
	    if (search!=null) { search.deregisterResultHandler(feedbackgazeTextArea); }
	}
	feedbackgazeTextArea = new NTextArea();
	feedbackgazeTextArea.setClock(niteclock);
	
	niteclock.registerTimeHandler((TimeHandler) feedbackgazeTextArea);
	if (search != null) { search.registerResultHandler(feedbackgazeTextArea); }
	
	List elist = null;
	
	try {
	    elist = searchEngine.search(
					nom,
					"($a feedbackgaze)");
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (elist == null || elist.size() == 1) {
	    System.err.println("NO FEEDBACKGAZE IN CORPUS!");
	    System.exit(0);
	}
	
	Iterator elit = elist.iterator();
	
	/* The first thing on the list returned by the search engine
	 * is a duff entry containing the names of the variables for
	 * the remaining things on the list.  Use the boolean to get
	 * past that entry.  */
	boolean first = true;
	while (elit.hasNext()) {
	    List reslist = (List) elit.next();
	    if (first) {
		first = false;
		continue;
	    }
	    
	    NOMWriteElement nwe = (NOMWriteElement) reslist.get(0);
	    showFeedbackGaze(nwe);
	    NTextElement nte = new NTextElement("\n", null);
	    NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	    nte.setDataElement((ObjectModelElement) nome);
	    feedbackgazeTextArea.addElement(nte);
	}
	
	/* top (or left) of the desktop area, containing the text */
	pane = new JScrollPane(feedbackgazeTextArea);
	iframe.getContentPane().add(pane);
	iframe.setVisible(true);
	desktop.add(iframe);
    }
    
    private void addTextAreaFeedbackContent() {
	iframe =
	    new JInternalFrame("Feedback Display", true, false, true, true);
	iframe.setSize(new Dimension(200, 160));
	iframe.setLocation(new Point(440, 400));
	if (feedbackTextArea != null) {
	    niteclock.deregisterTimeHandler((TimeHandler) feedbackTextArea);
	    if (search!=null) { search.deregisterResultHandler(feedbackTextArea); }
	}
	feedbackTextArea = new NTextArea();
	feedbackTextArea.setClock(niteclock);
	
	niteclock.registerTimeHandler((TimeHandler) feedbackTextArea);
	if (search != null) { search.registerResultHandler(feedbackTextArea); }
	
	List elist = null;
	
	try {
	    elist =
		searchEngine.search(
				    nom,
				    "($a feedback)");
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (elist == null || elist.size() == 1) {
	    System.err.println("NO FEEDBACK IN CORPUS!");
	    System.exit(0);
	}
	
	Iterator elit = elist.iterator();
	
	/* The first thing on the list returned by the search engine
	 * is a duff entry containing the names of the variables for
	 * the remaining things on the list.  Use the boolean to get
	 * past that entry.  */
	boolean first = true;
	while (elit.hasNext()) {
	    List reslist = (List) elit.next();
	    if (first) {
		first = false;
		continue;
	    }
	    
	    NOMWriteElement nwe = (NOMWriteElement) reslist.get(0);
	    showFeedback(nwe);
	    NTextElement nte = new NTextElement("\n", null);
	    NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	    nte.setDataElement((ObjectModelElement) nome);
	    feedbackTextArea.addElement(nte);
	}
	
	/* top (or left) of the desktop area, containing the text */
	pane = new JScrollPane(feedbackTextArea);
	iframe.getContentPane().add(pane);
	iframe.setVisible(true);
	desktop.add(iframe);
    }
    
    private void addTextAreaRouteGazeContent() {
	iframe =
	    new JInternalFrame("RouteGaze Display", true, false, true, true);
	iframe.setSize(new Dimension(220, 320));
	iframe.setLocation(new Point(220, 0));
	if (routegazeTextArea != null) {
	    niteclock.deregisterTimeHandler((TimeHandler) routegazeTextArea);
	    if (search!=null) { search.deregisterResultHandler(routegazeTextArea); }
	}
	routegazeTextArea = new NTextArea();
	routegazeTextArea.setClock(niteclock);
	
	niteclock.registerTimeHandler((TimeHandler) routegazeTextArea);
	if (search != null) { search.registerResultHandler(routegazeTextArea); }
	
	List elist = null;
	
	try {
	    elist =
		searchEngine.search(
				    nom,
				    "($a routegaze)");
	} catch (Throwable e) {
	    e.printStackTrace();
			System.exit(0);
	}
	if (elist == null || elist.size() == 1) {
	    System.err.println("NO ROUTEGAZE IN CORPUS!");
	    System.exit(0);
	}
	
	Iterator elit = elist.iterator();
	
	/* The first thing on the list returned by the search engine
	 * is a duff entry containing the names of the variables for
	 * the remaining things on the list.  Use the boolean to get
	 * past that entry.  */
	boolean first = true;
	while (elit.hasNext()) {
	    List reslist = (List) elit.next();
	    if (first) {
		first = false;
		continue;
	    }
	    
	    NOMWriteElement nwe = (NOMWriteElement) reslist.get(0);
	    showRouteGaze(nwe);
	    NTextElement nte = new NTextElement("\n", null);
	    NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	    nte.setDataElement((ObjectModelElement) nome);
	    routegazeTextArea.addElement(nte);
	}
	
	/* top (or left) of the desktop area, containing the text */
	pane = new JScrollPane(routegazeTextArea);
	iframe.getContentPane().add(pane);
	iframe.setVisible(true);
	desktop.add(iframe);
    }
    
    
    private void addTextAreaTranscriptionContent() {
	iframe = new JInternalFrame("Transcription Display", true, false, true, true);
	iframe.setSize(new Dimension(440, 240));
	iframe.setLocation(new Point(0, 320));
	if (transcriptionTextArea != null) {
	    niteclock.deregisterTimeHandler((TimeHandler) transcriptionTextArea);
	    if (search!=null) { search.deregisterResultHandler(transcriptionTextArea); }
	}
	transcriptionTextArea = new NTextArea();

	transcriptionTextArea.setClock(niteclock);
	niteclock.registerTimeHandler((TimeHandler) transcriptionTextArea);
	if (search != null) { search.registerResultHandler(transcriptionTextArea); }

	List outerelist = null;

	try {
	    outerelist = searchEngine.search(nom, "($a move)");
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (outerelist == null || outerelist.size() == 1) {
	    System.err.println("NO MOVES IN DATA!");
	    System.exit(0);
	}
	
	for (int i=1; i< outerelist.size(); i++) {
	    List outerreslist = (List) outerelist.get(i);
	    NOMWriteElement outernwe = (NOMWriteElement) outerreslist.get(0);
	    showMove(outernwe);
	    transcriptionTextArea.addElement(new NTextElement(": ", null));
	    showWords(outernwe);
	    transcriptionTextArea.addElement(new NTextElement("\n", null));
	}
	pane = new JScrollPane(transcriptionTextArea);
	iframe.getContentPane().add(pane);
	iframe.setVisible(true);
	desktop.add(iframe);
    }

    /* show the words in the corpus, recursively descending through
       the syntax layers and checking for markables at each stage. */
    private void showWords(NOMElement nwe) {
	if (nwe==null || nwe.getChildren()==null) { return; }
	for (Iterator kit=nwe.getChildren().iterator(); kit.hasNext(); ) {
	    NOMElement kid=(NOMElement)kit.next();
	    if (kid.getName().equalsIgnoreCase("word")) {
		NOMObjectModelElement nome= new NOMObjectModelElement(kid);
		transcriptionTextArea.addElement(new NTextElement(kid.getText() + " ", null,  nome));
	    } else {
		showWords(kid);
	    }
	}
    }

    private void showMove(NOMWriteElement nwe) {
	if (nwe == null) {
	    return;
	}
	String text = "";
	NOMAttribute label = nwe.getAttribute("type");
	if (label != null) {
	    text = label.getStringValue();
	}
	
	NTextElement nte = new NTextElement(text, null, nwe.getStartTime(), nwe.getEndTime());
	//	System.out.println("Add move start: " + nwe.getStartTime() + "; end: " + nwe.getEndTime());
	NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	nte.setDataElement((ObjectModelElement) nome);
	transcriptionTextArea.addElement(nte);
    }
    
    private void showFeedbackGaze(NOMWriteElement nwe) {
	if (nwe == null) {
	    return;
	}
	String text = nwe.getID() + ":" + nwe.getAttribute("label").getStringValue();
	NTextElement nte =
	    new NTextElement(text, null, nwe.getStartTime(), nwe.getEndTime());
	NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	nte.setDataElement((ObjectModelElement) nome);
	feedbackgazeTextArea.addElement(nte);
    }
    
    private void showFeedback(NOMWriteElement nwe) {
	if (nwe == null) {
	    return;
	}
	String text = nwe.getID() + ":" + nwe.getAttribute("label").getStringValue();
	NTextElement nte =
	    new NTextElement(text, null, nwe.getStartTime(), nwe.getEndTime());
	NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	nte.setDataElement((ObjectModelElement) nome);
	feedbackTextArea.addElement(nte);
    }
    
    private void showRouteGaze(NOMWriteElement nwe) {
	if (nwe == null) {
	    return;
	}
	String text = nwe.getID() + ":" + nwe.getAttribute("label").getStringValue();
	NTextElement nte =
	    new NTextElement(text, null, nwe.getStartTime(), nwe.getEndTime());
	NOMObjectModelElement nome = new NOMObjectModelElement(nwe);
	nte.setDataElement((ObjectModelElement) nome);
	routegazeTextArea.addElement(nte);
    }    
    
    private void showTree(NOMElement source_element, NTreeNode tree_node) {
	
	if (source_element == null) {
	    return;
	}
	NTreeNode newnode =
	    new NTreeNode(source_element.getAttribute("name").getStringValue());
	tree_node.addNode(newnode);
	if (source_element.getChildren() == null) {
	    return;
	}
	for (Iterator kit = source_element.getChildren().iterator();
	     kit.hasNext();
	     ) {
	    NOMElement kid = (NOMElement) kit.next();
	    showTree(kid, newnode);
	}
    }
    
    /**
     * We want to load one observation worth of data; return the
     * observation with the name
     * @param observation.  Choke if there isn't one.  */
    private NiteObservation observationWithName(String observation) {
	boolean found = false;
	NiteObservation returnval = null;
	List obs = controlData.getObservations();
	Iterator obs_it = obs.iterator();
	while ((obs_it.hasNext()) && (found == false)) {
	    NiteObservation next = (NiteObservation) obs_it.next();
	    if (observation.equals(next.getShortName()) == true) {
		found = true;
		returnval = next;
	    };
	}
	if (found == false) {
	    System.out.println(
			       "Observation named "
			       + observation
			       + " doesn't exist:  exiting...");
	    System.exit(0);
	}
	return returnval;
    }
    
    /**
     * Called to start the  application.
     * Legal command line arguments are:
     *<ul>
     *<li> -corpus  corpus </li>
     * <li> -observation observation </li>
     *</ul>
     *
     */
    public static void main(String[] args) {
	String corpus=null;
	String observation=null;
	
	if (args.length < 2 || args.length > 4) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-observation") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		observation=args[i];
	    } else {
		usage();
	    }
	}
	if (corpus == null) { usage(); }
	
	MonitorDisplay m = new MonitorDisplay(corpus, observation);
    }
    
    private static void usage() {
	System.err.println("Usage: java MonitorDisplay -c metadata-filename -o observation-name");
	System.exit(0);
    }

    /** A very simple "exit" action */
    public class ExitAction extends AbstractAction {
	public ExitAction() {
	    super("Exit");
	}
	public void actionPerformed(ActionEvent ev) {
	    System.exit(0);
	}
    }

    /** A "search" action - pops up a search GUI */
    public class SearchAction extends AbstractAction {
	public SearchAction() {
	    super("Search...");
	}
	public void actionPerformed(ActionEvent ev) {
	    search.popupSearchWindow();
	}
    }


}
