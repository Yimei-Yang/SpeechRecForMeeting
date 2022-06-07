import java.io.*;

import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.search.*;
import net.sourceforge.nite.nom.NOMException;
import net.sourceforge.nite.util.SearchResultTimeComparator;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.Hashtable;
import java.util.Iterator;


/**
 * SortedOutput is a utility for outputting tab-delimited data.  It
 * takes all elements resulting from the first matched variable of a
 * query, as long as they are timed, and put them in order of start
 * time.  Then it outputs one line per element the values of a the
 * named attributes with a tab character between each attribute.
 * 
 * To call the utility, which is in builds after NXT 1.3.2 but otherwise
 * available from CVS, set the classpath in the usual way, and then use 
 * 
 * java SortedOutput -c metadata_file_name -ob observation_name -q query -t -atts [attname+]
 * 
 * -corpus is required; it is the path to the metadata file.
 * 
 * -observation is optional: it's the name of the observation. If not
 * present the program cycles through all observations.required;
 * 
 * -query is required; the first matched variable from every result
 *   forms the basis of the output.
 *
 * -t is optional - if present the textual content of the element (or
    any subelements) is output *after* the attributes for the
    element. Newlines and tabs are replaced with a space.
 * 
 * -atts is required; input is expected as a space separated list of
    attributes. Note that if the attribute does now exist for some
    matched elements, a blank tab-stop will be the output.
 * 
 * java SortedOutput -c METADATA -ob OBS -q '($m move)' -t -atts type nite:start nite:end
 * 
 * will output a sorted list of moves for the observation consisting
 * of type attribute, start and end times, and after all th
 *
 * @author Jean Carletta, Jonathan Kilgour Feb 2005
 **/

public class SortedOutput { 
    
    // the data 
    private NOMWriteCorpus nom;
    private NiteMetaData meta;

    // the arguments passed in
    String corpusname;
    String observationname;
    String query;
    boolean text;
    List atts;

    private Engine searchEngine = new Engine();
    private Comparator mycomp = new SearchResultTimeComparator();

    public SortedOutput (String c, String o, String q, boolean t, List a) {

	corpusname = c;
	observationname = o;
	query = q;
	atts = a;
	text = t;

	try {
	    meta = new NiteMetaData(corpusname);
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}

	try {
	    // second arg sends log messages to System.err, not System.out.
	    nom = new NOMWriteCorpus(meta, System.err);
	    if (observationname!=null) {
		NObservation ob = meta.getObservationWithName(observationname);
		if (ob==null) {
		    System.err.println("No observation called '" + observationname + "' exists!");
		    System.exit(0);		    
		}
		nom.loadData(ob);
		searchAndPrint(nom, observationname, query, text, atts);
	    } else { // do for each observation in turn
		List obslist = meta.getObservations();
		for (int i = 0; i < obslist.size(); ++i) {
		    NiteObservation nobs = (NiteObservation) obslist.get(i);
		    nom.loadData(nobs);
		    searchAndPrint(nom, nobs.getShortName(), query, text, atts);
		    // clearData reloads the corpus resources, which is a bit strange.
		    nom.clearData();
		};
	    }
	} catch (NOMException nex) {
	    nex.printStackTrace();
	}
    }

    /** do the main work */
    private void searchAndPrint(NOMCorpus nom, String observationname, String q, boolean t, List atts) {
	try {
	    List reslist = searchEngine.search((SearchableCorpus)nom, q);
	    if (reslist.size()>0) {
		reslist.remove(0);
		Collections.sort(reslist, mycomp);
	    }
	    printResults(observationname, reslist, t, atts);
	} catch (Throwable ex) {
	    ex.printStackTrace();
	    System.exit(0);
	}
    }

    private void printResults(String ob, List reslist, boolean textout, List atts) {
	System.out.println("Observation: " + ob + "; Result size: " + reslist.size() + "; atts: " + atts.size());
	for (int i=0; i< reslist.size(); i++) {
	    List outerreslist = (List) reslist.get(i);
	    NOMElement el = (NOMElement) outerreslist.get(0);
	    if (el==null) { continue; }
	    for (int j=0; j<atts.size(); j++) {
		String attr = (String)atts.get(j);
        Comparable attrval = el.getAttributeComparableValue(attr);
		String val = "";
        try {
            val = (String)attrval;
        } catch (ClassCastException e) {
            val = ((Double)(attrval)).toString();
        }
        
		if (attr.equalsIgnoreCase(meta.getIDAttributeName())) {
		    val=el.getID();
		} else if (attr.equalsIgnoreCase(meta.getStartTimeAttributeName())) {
		    val=(new Double(el.getStartTime())).toString();
		} else if (attr.equalsIgnoreCase(meta.getEndTimeAttributeName())) {
		    val=(new Double(el.getEndTime())).toString();
		} else if (attr.equalsIgnoreCase(meta.getAgentAttributeName())) {
		    val=el.getAgentName();
		}
		if (val!=null) {
		    System.out.print(val);
		}
		System.out.print("\t");
	    }
	    if (textout) {
		String t = getTextRecursive(el);
		if (t!=null) {
		    t = t.replace('\n',' ');
		    System.out.print(t);
		}
	    }
	    System.out.println();
	}
	
    }

    /** return text of this element or any kids */
    private String getTextRecursive(NOMElement el) { 
	String ret="";
	if (el==null) { return ret; }
	List kids = el.getChildren();
	if (kids==null) {
	    String t = el.getText();
	    if (t!=null && t.trim().length()>0) { ret += t.trim() + " "; }
	} else {
	    for (Iterator kit=kids.iterator(); kit.hasNext(); ) {
		ret += getTextRecursive((NOMElement) kit.next());
	    }
	}
	return ret;
    }


    /**
     * Called to start the  application. 
     * Call like this:
     *    java SortedOutput -corpus metadata_file_name -observation observation_name -query query -atts attname1 attname2 .. 
     *
     * with corpus, query and atts required; observation optional; 
     */
    public static void main(String[] args) {
	String corpus=null;
	String observation=null;
	String query = null;
	boolean text=false;
	List atts = new ArrayList();

	if (args.length < 6) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-text") || flag.equals("-t")) {
		text=true;
	    } else if (flag.equals("-observation") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		observation=args[i];
	    } else if (flag.equals("-query") || flag.equals("-q")) {
		i++; if (i>=args.length) { usage(); }
		query=args[i];
	    } else if (flag.equals("-atts") || flag.equals("-a")) {
		int j=i+1;
		while (j<args.length) {
		    atts.add(args[j++]);
		}
		break;
	    } else {
		usage();
	    }
	}
	if ((corpus == null) || (query == null)) { usage(); }
	if (atts.size()==0) { usage(); }
	SortedOutput m = new SortedOutput(corpus, observation, query, text, atts);
    }

    private static void usage () {
	System.err.println("Usage: java SortedOutput -corpus <path-to-metadata> -observation <obsname> -query <query> -t -atts <attname1> <attname2> .. ");
	System.exit(0);
    }
}





