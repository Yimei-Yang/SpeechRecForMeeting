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

import java.lang.Math.*;

/**
 * Transform a NXT corpus into the Turtle compressed RDF format.
 *
 * @author Jonathan Kilgour July 2007
 **/
public class NXTtoRDF { 
    
    // the data 
    private NOMWriteCorpus nom;
    private NiteMetaData meta;

    // the arguments passed in
    String corpusfilename;
    String corpusname;
    String observationname;
    boolean orderedchildren=false;
    String spaces="    ";
    boolean donecorpuslevel=false;
    
    //private Engine searchEngine = new Engine();
    //private Comparator mycomp = new SearchResultTimeComparator();

    public NXTtoRDF (String c, String n, String o, boolean order) {
	corpusfilename = c;
	corpusname = n;
	observationname = o;
	orderedchildren=order;
       
	try {
	    meta = new NiteMetaData(corpusfilename);
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}

	try {
	    // second arg sends log messages to System.err, not System.out.
	    nom = new NOMWriteCorpus(meta, System.err);
	    nom.setLazyLoading(false);

	    printHeader(corpusname);

	    if (observationname!=null) {
		NObservation ob = meta.getObservationWithName(observationname);
		if (ob==null) {
		    System.err.println("No observation called '" + observationname + "' exists!");
		    System.exit(0);		    
		}
		nom.loadData(ob);
		printRDF(nom, observationname);
	    } else { // do for each observation in turn
		List obslist = meta.getObservations();
		for (int i = 0; i < obslist.size(); ++i) {
		    NiteObservation nobs = (NiteObservation) obslist.get(i);
		    nom.loadData(nobs);
		    printRDF(nom, nobs.getShortName());
		    donecorpuslevel=true;
		    nom.clearData();
		};
	    }
	} catch (NOMException nex) {
	    nex.printStackTrace();
	}
    }

    /** print some namespaces for a header - kind of hardwired at the moment. */
    private void printHeader (String corpusname) {
	System.out.println("@prefix "+corpusname+": <http://www."+corpusname+".org/corpus/> .");
	System.out.println("@prefix nxt: <http://www.ltg.ed.ac.uk/NITE/> .");
	System.out.println("@prefix "+corpusname+"schema: <http://www.ltg.ed.ac.uk/NITE/"+corpusname+"schema/> .");
	System.out.println("@prefix dada: <http://dada.org/schema/> .\n");
    }

    /** print a meeting worth of RDF */
    private void printRDF(NOMCorpus nom, String observationname) {
        List vars=null;
        List reslist=null; 
        Hashtable myvars = new Hashtable();
	for (Iterator nit=nom.NOMWalker(); nit.hasNext(); ) {
	    printRDF((NOMElement)nit.next());
	}
    }   

    private String replaceChars(String in, String old, String news) {
	if (in==null) { return null; }
	if (in.lastIndexOf(old)>-1) {
	    int ind=-1;
	    while ((ind=in.indexOf(old))>-1) {
		String star="";	if (ind!=0) { star=in.substring(0,ind); }
		in = star + news + in.substring(ind+1,in.length());
	    }
	}
	return in;
    }
    
    private String escape(String in) {
	return replaceChars(replaceChars(in,"\"",""), "\n", " ");
    }
    

    /** print an element worth of RDF */
    private void printRDF(NOMElement nel) {
	String pointstr="";
	String childstr="";

	// this makes sure we don't repeat ontologies etc.
	if (donecorpuslevel && !(nel instanceof NOMWriteAnnotation)) {
	    return;
	}

	System.out.println(corpusname+":"+nel.getID()+" a dada:annotation;");
	System.out.print(spaces+"nxt:type\t\""+corpusname+"schema:"+nel.getName()+"\"");
	if (nel.getText()!=null) {
	    System.out.print(";\n"+spaces+"nxt:content\t\""+escape(nel.getText())+"\"");
	}
	double st=nel.getStartTime();
	if (!Double.isNaN(st) && st!=NOMElement.UNTIMED) {
	    System.out.print(";\n"+spaces+"nxt:starttime\t\""+st+"\"");
	}
	double et=nel.getEndTime();
	if (!Double.isNaN(et) && et!=NOMElement.UNTIMED) {
	    System.out.print(";\n"+spaces+"nxt:endtime\t\""+et+"\"");
	}
	List atts = nel.getAttributes();
	if (atts!=null) {
	    for (Iterator ait=atts.iterator(); ait.hasNext(); ) {
		NOMAttribute att = (NOMAttribute)ait.next();
		if (att.getName().indexOf("xmlns:")<0) {
		    System.out.print(";\n"+spaces+"nxt:"+att.getName()+"\t\""+att.getComparableValue()+"\"");
		}
	    }
	}

	List kids = nel.getChildren();
	if (kids!=null && kids.size()>0) {
	    int childcount=1;
	    for (Iterator kit=kids.iterator(); kit.hasNext(); ) { 
		NOMElement kid = (NOMElement)kit.next();
		if (orderedchildren) {
		    String childid=nel.getID()+".child"+childcount;
		    System.out.print(";\n"+spaces+"nxt:child\t\""+corpusname+":"+childid+"\"");
		    childstr += corpusname+":"+childid+" a dada:annotation;\n";
		    childstr+=spaces+"nxt:type\t\""+corpusname+"schema:nxtchild\";\n";
		    childstr+=spaces+"nxt:parent\t\""+corpusname+":"+nel.getID()+"\";\n";
		    childstr+=spaces+"nxt:childid\t\""+corpusname+":"+kid.getID()+"\";\n";
		    childstr+=spaces+"nxt:order\t\""+childcount+"\" . \n";
		    childcount++;
		} else {
		    System.out.print(";\n"+spaces+"nxt:child\t\""+corpusname+":"+kid.getID()+"\"");
		}
	    }
	}

	List pointers = nel.getPointers();
	if (pointers!=null && pointers.size()>0) {
	    int pcount=1;
	    for (Iterator pit=pointers.iterator(); pit.hasNext(); ) {
		NOMPointer point = (NOMPointer)pit.next();
		String pointid=nel.getID()+".pointer"+pcount;
		pcount++;
		System.out.print(";\n"+spaces+"nxt:pointer\t\""+corpusname+":"+pointid+"\"");
		pointstr += corpusname+":"+pointid+" a dada:annotation;\n";
		pointstr+=spaces+"nxt:type\t\""+corpusname+"schema:nxtpointer\";\n";
		// next line is redundant..
		try {
		    pointstr+=spaces+"nxt:from\t\""+corpusname+":"+point.getFromElement().getID()+"\";\n";
		    pointstr+=spaces+"nxt:to\t\""+corpusname+":"+point.getToElement().getID()+"\";\n";
		} catch (Exception ex) {
		    System.err.println("Failed to resolve pointer from element " +nel.getID());
		}
		pointstr+=spaces+"nxt:role\t\""+corpusname+":"+point.getRole()+"\" . \n";
	    }
	}

	System.out.println(" . ");

	System.out.println(childstr+"\n"+pointstr);
    }

    /**
     * Called to start the  application. 
     * Call like this:
     *    java NXTtoRDF -corpus metadata_file_name -observation observation_name [ -orderchildren ] 
     *
     * with corpus required; observation optional; -orderchildren meaning ordered
     * children - this creates separate nodes for each parent/child
     * expressing the order.
     */
    public static void main(String[] args) {
	String corpus=null;
	String name=null;
	String observation=null;
	boolean ordered=false;
    
	if (args.length < 4) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-corpusname") || flag.equals("-n")) {
		i++; if (i>=args.length) { usage(); }
		name=args[i];
	    } else if (flag.equals("-observation") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		observation=args[i];
	    } else if (flag.equals("-orderchildren") || flag.equals("-oc")) {
		ordered=true;
	    } else {
		System.err.println("ERROR: Unknown option - " + flag);
		usage();
	    }
	}
	if (corpus==null || name==null) { System.err.println("ERROR: -corpus and -name arguments required."); usage(); }
        NXTtoRDF m = new NXTtoRDF(corpus,name,observation,ordered);
    }

    private static void usage () {
	System.err.println("Usage: java NXTtoRDF -corpus <path-to-metadata> -corpusname <name> [ -observation <obsname> ] [ -orderchildren ] ");
	System.exit(0);
    }
}





