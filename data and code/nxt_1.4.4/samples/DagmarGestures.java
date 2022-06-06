import java.util.*;
import java.io.File;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.Color;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.tree.*;
import javax.swing.plaf.metal.*;
import javax.swing.text.Style;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.NOMException;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.gui.textviewer.*;
import net.sourceforge.nite.gui.mediaviewer.*;
import net.sourceforge.nite.nxt.*;
import net.sourceforge.nite.nstyle.handler.*;
import net.sourceforge.nite.time.*;

/**
 * Simple program that shows Dagmar gestures with a video.
 * 
 * @author Jonathan Kilgour March 2003
 **/
public class DagmarGestures implements ActionListener {
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
    NTextArea nta;
    NTree ntree;
    JScrollPane pane;
    JDesktopPane desktop;
    NOMWriteElement changed=null;
    NTextElement high=null;
    Style bluestyle=null;
    String bluestylename="blue";
    String TYPE="type";
    
    private Engine searchEngine = new Engine();
    
    public DagmarGestures(String c, String o) {
	corpusname = c;
	observationname = o;
	
	try {
	    controlData = new NiteMetaData(corpusname);
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}
	
	niteclock = new DefaultClock();
	TextStyleHandler bluestylehandler = new TextStyleHandler();
	bluestylehandler.init("", null);
	bluestylehandler.setName("blue");
	bluestylehandler.makeNewStyle();
	bluestylehandler.setStyle( bluestylehandler.getColours(bluestylehandler.getStyle(), "blue", "blue"));
	bluestyle=bluestylehandler.getStyle();
	if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
	    System.out.println("This is a standoff corpus: NOM being loaded");
	    try {
		nom = new NOMWriteCorpus(controlData);
		NObservation obs = controlData.getObservationWithName(observationname);
		nom.loadData(obs);
                System.out.println("Finished loading.");
		setupInterface(nom);
	    } catch (NOMException nex) {
		nex.printStackTrace();
	    }
	} else {
	    System.out.println("This is a standalone or simple corpus: no NOM has been loaded");
	}

    }

    /** add a frame to the UI with a button for changing gesture type */
    private void addButtonFrame() {
	JInternalFrame iframe =
	    new JInternalFrame("Actions", true, false, true, true);
	iframe.setSize(new Dimension(280, 70));
	iframe.setLocation(new Point(450,0));

	JPanel panel = new JPanel();
	JButton nbut = new JButton("Change gesture type");
	nbut.addActionListener(this);
	panel.add(nbut);
	iframe.getContentPane().add(panel);
	iframe.setVisible(true);
	desktop.add(iframe);
    }

    /** set up the entire UI */
    private void setupInterface(NOMWriteCorpus nom) {
	frame = new JFrame();
        frame.setTitle("Dagmar Corpus Gesture Coder");
	JMenuBar menubar = new JMenuBar();
	JMenu file = new JMenu("File");
	file.add(new SaveAction());
	file.add(new ExitAction());
	menubar.add(file);
	frame.setJMenuBar(menubar);
	desktop = new JDesktopPane();

	addTextAreaContent();

	NSignal sig = (NSignal) controlData.getSignals().get(0); // first signal;
	if (sig.getMediaType()==NSignal.VIDEO_SIGNAL) {
	    String video_filename=controlData.getSignalPath() + File.separator +
		observationname + "." + sig.getName() + "." + sig.getExtension();
	    System.out.println("Video file: " + video_filename);
	    video = new NITEVideoPlayer(new File(video_filename), (DefaultClock) niteclock);
	    video.setLocation(new Point(0, 240));
	    video.setSize(new Dimension(450, 320));
	    niteclock.getDisplay().setLocation(new Point(0,560));
	    niteclock.getDisplay().setSize(new Dimension(450,150));
	    desktop.add(niteclock.getDisplay());
	    desktop.add(video);
	}

	addListContent();
	addButtonFrame();

	JPanel panel = new JPanel();
	desktop.setSize(new Dimension(740, 760));
	frame.getContentPane().add(desktop);
	frame.setSize(new Dimension(740, 760));
	frame.setVisible(true);
    }
        
    /** Add the gestures to the main text area */
    private void addTextAreaContent( ) {
	iframe =
	    new JInternalFrame("Text Display", true, false, true, true);
	iframe.setSize(new Dimension(450, 240));
	if (nta != null) {
	    niteclock.deregisterTimeHandler((TimeHandler)nta);	
	}
	nta = new NTextArea();
	//	nta.setClock(niteclock);
	niteclock.registerTimeHandler((TimeHandler)nta);	
	nta.addStyle(bluestylename, bluestyle);

	List elist = null;
	
	try {
	    elist = searchEngine.search(nom, "($a gesture)");
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (elist == null || elist.size() == 1) {
	    System.err.println("NO GESTURES IN CORPUS!");
	    System.exit(0);
	}

	/* Iterate through the results (the first result is just a
           list of variables) */
	Iterator elit = elist.iterator();
	boolean first = true;
	//nta.addElement(new NTextElement(".", null));
	while (elit.hasNext()) {
	    List reslist = (List) elit.next();
	    if (first) {
		first = false;
		continue;
	    }

	    NOMWriteElement nwe = (NOMWriteElement) reslist.get(0);
	    showGesture(nwe);
	    nta.addElement(new NTextElement("\n", null));
	}

	/* top (or left) of the desktop area, containing the text */
	pane = new JScrollPane(nta);
	iframe.getContentPane().add(pane);
	iframe.setVisible(true);
	desktop.add(iframe);
	if (high!=null) {
	    System.out.println("YES - highlighting");
	    nta.setHighlighted(NTextArea.SELECTION_HIGHLIGHTS, high);
	    //nta.setHighlighted(high, MetalLookAndFeel.getTextHighlightColor());
	}
    }

    /** Display an individual gesture on the text area display */
    private void showGesture(NOMWriteElement nwe) {
	if (nwe==null) { return; }
	NOMPointer point = (NOMPointer) nwe.getPointerWithRole(TYPE);

	String text = "Gesture " + nwe.getID() + ":";
	String text2 = " type: ";
	if (point!=null && point.getRole().equalsIgnoreCase(TYPE)) {
	    NOMElement pel = point.getToElement();
	    text2 += pel.getAttribute("name").getStringValue();
	}
	    
	NOMObjectModelElement nome= new NOMObjectModelElement(nwe);
	NTextElement nte = new NTextElement(text, bluestylename,
				    nwe.getStartTime(), nwe.getEndTime(), nome);
	nta.addElement(nte);

	NTextElement nte2 = new NTextElement(text2, null,
				    nwe.getStartTime(), nwe.getEndTime(), nome);
	nta.addElement(nte2);
	if (nwe==changed) {
	    high=nte; // store this as the thing to be highlighted
	}
    }

    /** Display the hierarchy of gesture types */
    private void addListContent( ) {
	iframe2 =
	    new JInternalFrame("Gesture Hierarchy", true, false, true, true);
	iframe2.setSize(new Dimension(280, 530));
	iframe2.setLocation(new Point(450, 70));
	List elist = null;
	try {
	    elist = searchEngine.search(nom, "($a gtype)");
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (elist == null || elist.size() == 1) {
	    System.err.println("NO GESTURE TYPES IN CORPUS!");
	    System.exit(0);
	}

	NOMElement gesture_top = (NOMElement) ((ArrayList)elist.get(1)).get(0);
	NTreeNode root = new NTreeNode(gesture_top.getAttribute("name").getStringValue());
	showTree(gesture_top, root);

	ntree = new NTree(root);
	// Open all the root's children
	Enumeration kiden=root.children();
	while (kiden.hasMoreElements() ) {
	    DefaultMutableTreeNode jtn=(DefaultMutableTreeNode)kiden.nextElement();
	    TreeNode ktn=null;
	    if (jtn.getChildCount()!=0) { ktn=jtn.getFirstChild(); }
	    if (ktn==null) { continue; }
	    TreePath path = new TreePath(((DefaultTreeModel) ntree.getModel()).getPathToRoot(ktn));
	    if (!ntree.isVisible(path)) {
		ntree.scrollPathToVisible(path);
	    }
	}
	/* top (or left) of the desktop area, containing the text */
	pane = new JScrollPane(ntree);
	iframe2.getContentPane().add(pane);
	iframe2.setVisible(true);
	desktop.add(iframe2);
    }

    private void showTree(NOMElement source_element, NTreeNode tree_node) {
	if (source_element==null) { return; }
	NTreeNode newnode=null;
	if (source_element.getParents()!=null) {
	    newnode = new NTreeNode(source_element.getAttribute("name").getStringValue());
	    NOMObjectModelElement nome= new NOMObjectModelElement(source_element);
	    newnode.setDataElement((ObjectModelElement)nome);
	    tree_node.addNode(newnode);
	} else {
	    newnode=tree_node;
	}
	if (source_element.getChildren() == null) { return; }
	for (Iterator kit=source_element.getChildren().iterator(); kit.hasNext(); ) { 
	    NOMElement kid = (NOMElement) kit.next();
	    showTree(kid, newnode);
	}
    }
    
    
    /**
     * Called to start the  application.
     * Legal command line arguments are:
     *<ul>
     *<li> corpus </li>
     * <li> observation </li>
     *</ul>
     *
     */
    public static void main(String[] args) {
	if (args.length != 2) {
	    usage();
	}
	String c = args[0];
	String o = args[1];
	
	DagmarGestures m = new DagmarGestures(c, o);
    }
    
    private static void usage() {
	System.err.println("Usage: java DagmarGestures metadata-filename observation-name");
	System.exit(0);
    }

    /** respond to button presses etc (ActionListener interface) */
    public void actionPerformed(ActionEvent ae) {
	//desktop.remove(iframe);
	Set sel = nta.getHighlightedTextElements(NTextArea.SELECTION_HIGHLIGHTS);
	if (sel == null || sel.size()==0) {
	    System.out.println("Nothing selected in gesture list");
	    return;
	} 
	Set sel2 = ntree.getSelectedTreeNodes();
	if (sel2 == null || sel2.size()==0) {
	    System.out.println("Select something in gesture hierarchy");
	    return;
	}
	NTreeNode ntn= (NTreeNode)sel2.iterator().next();
	if (ntn==null) { 
	    System.out.println("Select something in gesture hierarchy");
	    return; 
	}

	
	Iterator sit = new ArrayList(sel).iterator();
	while (sit.hasNext()) {
	    Object ob = sit.next();
	    if (ob==null) { continue; }
	    NTextElement nte = (NTextElement) ob;
	    NOMObjectModelElement nome = (NOMObjectModelElement)nte.getDataElement();
	    if (nome==null) { continue; }
	    changed= (NOMWriteElement)nome.getElement();
	    if (changed==null) { continue; }
	    try {
		NOMPointer point = (NOMPointer) changed.getPointerWithRole(TYPE);
		if (point==null) {
		    //System.out.println("Add a new pointer!");
		    NOMObjectModelElement to=(NOMObjectModelElement)ntn.getDataElement();
		    point = new NOMWritePointer(nom,TYPE,changed,to.getElement());
		    changed.addPointer(point);
		} else {
		    NOMObjectModelElement to=(NOMObjectModelElement)ntn.getDataElement();
		    point.setToElement(to.getElement());
		}
		if (nte.getText().indexOf("type:")>=0) {
		    String val = " type: " + (String)point.getToElement().getAttributeComparableValue("name");
		    NTextElement nte2 = new NTextElement(val, null, changed.getStartTime(), changed.getEndTime(), nome);
		    nta.redisplayTextElement(nte, nte2);
		} else {
		    nta.removeHighlighted(NTextArea.SELECTION_HIGHLIGHTS, nte);
		}
	    } catch (NOMException nex) {
		nex.printStackTrace();
	    }
	}
    }


    /** Save the NOM */
    public class SaveAction extends AbstractAction {
	public SaveAction() {
	    super("Save corpus");
	}
	public void actionPerformed(ActionEvent ev) {
	    try {
		controlData.writeMetaData(corpusname);
		nom.serializeCorpus();
	    } catch (Exception e) {
		e.printStackTrace();
	    }
	}
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
}
