import java.awt.Dimension;
import java.awt.Color;
import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.*;
import java.util.logging.Logger;

import net.sourceforge.nite.gui.timelineviewer.*;
import net.sourceforge.nite.gui.transcriptionviewer.TranscriptionToTextDelegate;
import net.sourceforge.nite.gui.util.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.nom.*;
import net.sourceforge.nite.util.*;

public class SimpleTimelineDisplay extends AbstractCallableTool {
    private NiteTimeline timeline;
    private int timescale = 100;
    private String annotatorattr = "coder";
    private String pathtoannotators = ".";
    private GenericNOMElementViewer view;
	
    //TODO: Add command line descriptions for these
	
    // Name of NOMElements to be displayed
    private String displayelementname = "dact";
    // Name of layer (defines grouping of NOMElements)
    private String layerattributename = null; //"who";
    // Which text to display as label - null means getText()
    private String textattributename = null;
    // Coding name of elements created on the timeline
    private String newelementname = null;
    // Agent of elements created on the timeline
    private String newelementagent = null;
    // Which attribute of pointed-to element to display as text
    private String pointerattributename = null;


    public void addTimedResults(List queryresults) 
    {
	System.out.println("addTimedResults: " + queryresults.size() + " results found.");
	for(Iterator qit = queryresults.iterator(); qit.hasNext(); )
	    {
		List nextresult = (List) qit.next();
		if (nextresult.get(0) instanceof String) 
		    {
			continue;
		    }
		for (Iterator eit = nextresult.iterator(); eit.hasNext(); )
		    {
			Object nextitem = eit.next();
			if (nextitem instanceof List) 
			    {
				addTimedResults((List) nextitem);
			    }
			else 
			    {
				NOMElement ne = (NOMElement) nextitem;
				if (ne.getStartTime() == NOMElement.UNTIMED || ne.getEndTime() == NOMElement.UNTIMED) 
				    {
					continue;
				    }

				//System.out.println("Adding element: " + ne);
				timeline.addElement(ne);
			    }

		    }	     
	    }
    }

    private String printAttributes(List atts) {
	String out = "";
	Iterator it = atts.iterator();
	while (it.hasNext()) {
	    NOMAttribute a = (NOMAttribute) it.next();
	    out += a.getName() + " : " + a.getStringValue() + ", ";
	}
	return "[" + out + "]";
    }

    private String ZOOM_IN_ACTION = "Zoom in on timescale";
    private String ZOOM_OUT_ACTION = "Zoom out of timescale";
    private String ZOOM_WINDOW_ACTION = "Fit timescale to window";

    public class zoomAction extends AbstractAction 
    {
	protected double factor;

	public zoomAction(double zoomFactor) 
	{
	    super("Zoom " + (zoomFactor > 1 ? "in" : "out") );
	    factor = zoomFactor;
	};

	public void actionPerformed(ActionEvent ev) 
	{
	    TimeGrid tg = timeline.getGrid();
	    timescale *= factor;
	    // If timescale is not an integer less than fit-to-window size
	    if (timescale <= 1 || timescale >= tg.getMaxTimeInMilliseconds() / tg.getParent().getWidth()) {
		timescale /= factor;
	    }
	    tg.setScale(timescale);
	    tg.repaint(tg.getBounds(null));
	    System.out.println("" + factor + " * zoom : scale is now " + timescale); 
	};
    }

    /* zoom to a fixed factor */
    public class fixedZoomAction extends AbstractAction 
    {
	public fixedZoomAction() 
	{
	    super("Fit to window.");
	}

	public void actionPerformed(ActionEvent ev)
	{
	    TimeGrid tg = timeline.getGrid();
	    // Problem here is that tg.getWidth() depends on initial scale... :-(
	    timescale = (int) (tg.getMaxTimeInMilliseconds() / tg.getParent().getWidth());
	    tg.setScale(timescale);
	    tg.repaint(tg.getBounds(null));
	    System.out.println("Zoom : fit to window.");
	}

    }


    protected void addActions() 
    {
	Action act = null;

	act = new zoomAction(2.0); 
	act.putValue(Action.SHORT_DESCRIPTION,"Show fewer events in the timeline.");
	act.putValue(Action.MNEMONIC_KEY,new Integer(java.awt.event.KeyEvent.VK_PLUS));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_PLUS, java.awt.event.InputEvent.CTRL_MASK ));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_ADD, java.awt.event.InputEvent.CTRL_MASK ));
	getActionMap().put(ZOOM_IN_ACTION, act);

	act = new zoomAction(0.5); 
	act.putValue(Action.SHORT_DESCRIPTION,"Show more events in the timeline.");
	act.putValue(Action.MNEMONIC_KEY,new Integer(java.awt.event.KeyEvent.VK_MINUS));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_MINUS, java.awt.event.InputEvent.CTRL_MASK ));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_SUBTRACT, java.awt.event.InputEvent.CTRL_MASK ));
	getActionMap().put(ZOOM_OUT_ACTION, act);

	act = new fixedZoomAction(); 
	act.putValue(Action.SHORT_DESCRIPTION,"Show all events in the timeline.");
	act.putValue(Action.MNEMONIC_KEY,new Integer(java.awt.event.KeyEvent.VK_ASTERISK));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_ASTERISK, java.awt.event.InputEvent.CTRL_MASK ));
	act.putValue(Action.ACCELERATOR_KEY,KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_MULTIPLY, java.awt.event.InputEvent.CTRL_MASK ));
	getActionMap().put(ZOOM_WINDOW_ACTION, act);

    }

    private void addMenus() 
    {
	addActions();

	JMenu zoomM = new JMenu("Zoom");
	getMenuBar().add(zoomM);
	getMenuMap().put("Zoom",zoomM);
	zoomM.add(new JMenuItem(getActionMap().get(ZOOM_IN_ACTION)));
	zoomM.add(new JMenuItem(getActionMap().get(ZOOM_OUT_ACTION)));
	zoomM.add(new JMenuItem(getActionMap().get(ZOOM_WINDOW_ACTION)));
    }	

    public SimpleTimelineDisplay(String [] args) {
	parseArguments(args);
		
	for (int i = 0; i < args.length; ++i) {
	    if (args[i].equalsIgnoreCase("-displayelement")) {
		displayelementname = args[++i];
	    } else if (args[i].equalsIgnoreCase("-layerattribute")) {
		layerattributename = args[++i];
	    } else if (args[i].equalsIgnoreCase("-textattribute")) {
		textattributename = args[++i];
	    } else if (args[i].equalsIgnoreCase("-pointerattribute")) {
		pointerattributename = args[++i];
	    } else if (args[i].equalsIgnoreCase("-newelementname")) {
		newelementname = args[++i];
	    } else if (args[i].equalsIgnoreCase("-newelementagent")) {
		newelementagent = args[++i];
	    }
	}
		
	initConfig();
	initializeCorpus(getCorpusName(),getObservationName()); //, getAnnotatorName()); //initialize the corpus, given the settings passed as arguments
	setupMainFrame(getConfig().getApplicationName()); //setup a main frame 
	initLnF(); // Look and feel. Can be ignored/left out if you happen to like the default metal look-and-feel
	setupDesktop(); //setup a desktop on which all the windows (media viewer, transcription view, etc) are added

	setupLog(Logger.global, 530, 535, 465, 90); //a logging window, useful for giving feedback to the user
	setupMediaPlayer(695,15,300,400); //a mediaplayer: necessary if you want video or audio players synchronized to the transcription view

	setupSearch();
	setupMenus();
	addMenus();

	timeline = new NiteTimeline("Testing Time Trial", getClock());

	//TODO: Make scale interactively setable
	timeline.getGrid().setScale(timescale);
	timeline.getGrid().setBlobLabel(true);
		
	if (newelementname != null && newelementagent != null) {
	    NElementCreator nec = new NElementCreator(getCorpus());
	    nec.forceName(newelementname);
	    nec.forceAgent(newelementagent);
	    timeline.getGrid().setElementCreator(nec);
	    timeline.getGrid().setSnapToTime(false);
	} else {
	    System.out.println("INFO: NOMElement creation disabled.\nINFO: Use -newelementname and -newelementagent to enable.");
	}
		
	getQueryHandler().registerResultHandler(timeline.getGrid());
		
	// this layer to text delegate simply separates layers by
	// element name. Note that unless we separate depth on agent,
	// this would be likely to collapse things on top of each
	// other (also compounded by resources).
	TranscriptionToTextDelegate nameOfElement = 
	    new TranscriptionToTextDelegate() {

		public String getTextForTranscriptionElement(NOMElement nme) {
		    return nme.getName();
		}
				
	    };

	// this layer to text delegate is often more useful:
	// separates layers by metadata layer name; agent and
	// resource.
	TranscriptionToTextDelegate colourOfElement = 
	    new TranscriptionToTextDelegate() {

		public String getTextForTranscriptionElement(NOMElement nme) {
		    String rstr = nme.getName();
		    String ln=null;
		    try { ln = nme.getLayer().getName(); } 
		    catch (Exception ex) { }
		    if (ln!=null) { rstr=ln; }
		    String a = nme.getAgentName();
		    if (a!=null) { rstr += ": "+ a; }
		    NResource nr = nme.getResource();
		    if (nr!=null) { rstr += " ("+ nr.getID() + ")"; }
		    return rstr + "  ";
		}
				
	    };

	if (layerattributename == null) {
	    timeline.setLayerTextDelegate(colourOfElement);
	} else {
	    ElementToTextDelegate eetd = timeline.getElementToLayerDelegate();
	    eetd.setTranscriptionAttribute(layerattributename);
	}

	// This element would give different depths to agents in the same layer
        ElementToLayerDepthDelegate separateAgentByDepth = 
            new ElementToLayerDepthDelegate() {
                public int getElementLayerDepth(NOMElement nme) {
                    return getMetaData().getAgents().indexOf(nme.getAgent()) + 2; //index from 1 (no agent)
                }
	    };

	// but we decided it's clearer to have agents with separate
	// layer names for now..
        ElementToLayerDepthDelegate separateByRecursiveDepth = 
            new ElementToLayerDepthDelegate() {
                public int getElementLayerDepth(NOMElement nme) {
                    return nme.getRecursiveDepth()+1;
                }
	    };

        timeline.setElementToLayerDepthDelegate(separateByRecursiveDepth);

	// this delegate just prints IDs on blobs - not very useful
	TranscriptionToTextDelegate idOfElement = 
	    new TranscriptionToTextDelegate() {

		public String getTextForTranscriptionElement(NOMElement nme) {
		    return nme.getID();
		}
				
	    };

	// this delegate prints the value of a particular attribute of
	// an element we point to - could be useful for pointers into
	// ontologies.
	TranscriptionToTextDelegate pointerAttrOfElement = 	new TranscriptionToTextDelegate() {
		
		public String getTextForTranscriptionElement(NOMElement nme) {
		    List ps=nme.getPointers();
		    if(ps==null) { return ""; }
		    for (Iterator pit=ps.iterator(); pit.hasNext(); ) {
			try {
			    NOMPointer np = (NOMPointer)pit.next();
			    NOMElement nel = np.getToElement();
			    String val = (String)nel.getAttributeComparableValue(pointerattributename);
			    if (val!=null) { 
				return val;
			    }
			} catch (Exception ex) {
			}
		    }
		    return "";
		}
	    };

	// Having declared some alternatives, now decide which
	// delegate to use for displaying text on each blob...
	if (textattributename != null) {
	    timeline.setBlobTextAttribute(textattributename);
	} else if (pointerattributename!=null) {
	    timeline.setBlobTextDelegate(pointerAttrOfElement);
	} else {
	    timeline.setBlobTextDelegate(idOfElement);
	}

	addTimedResults(search("($a " + displayelementname + ")"));

	view = new GenericNOMElementViewer();
	view.setLocation(700, 0);
	view.setSize(200, 600);
	timeline.getGrid().addNOMElementSelectionListener(view);

	//addTimedResults(search("($w w)"));

	//addTimedResults(search("($v posture|head|hand|foa|movement|named-entity|aseg)"));

	//timeline.getGrid().testCustomLayer();
        //timeline.getGrid().addCustomLayer(new TimeLayer(0, timeline.getGrid().getMaxTimeInMilliseconds(), timeline.getGrid(), null));
       
	getDesktop().add(view);
	view.show();

	timeline.getFrame().setSize(new Dimension(700, 600));
	getDesktop().add(timeline.getFrame());
	Debug.print("Finished loading data.");
    }

    protected void initNomAnnotatorSpecificLoads(NOMWriteCorpus nom) 
    {
	return;
    }


    public static void main(String[] args) {
	SimpleTimelineDisplay nt = new SimpleTimelineDisplay(args);
    }

}
