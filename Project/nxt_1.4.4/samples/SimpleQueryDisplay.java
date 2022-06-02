import java.util.*;
import java.io.*;

// Import NITE stuff
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.NOMException;
import net.sourceforge.nite.search.*;

/**
 * Takes a corpus metadata file and a NQL query as arguments and
 * displays the results in a (Swing) JTree
 * 
 * @author Jonathan Kilgour, Feb 2003
 **/

public class SimpleQueryDisplay { 
    private Engine searchEngine = new Engine();

    public SimpleQueryDisplay(String c, String q) {
	NiteMetaData meta;
	NOMWriteCorpus nom;

	try {
	    meta = new NiteMetaData(c);
	    nom = new NOMWriteCorpus(meta);
	    nom.loadData();
	    List elist = searchEngine.search(nom, q);
	    displayResults(elist);
	} catch (NiteMetaException nme) {
	    System.err.println("Failed to load metadata file " + c);
	    System.exit(0);
	} catch (NOMException nex) {
	    System.err.println("NOM ERROR ");
	    nex.printStackTrace();
	    System.exit(0);
	} catch (Throwable ex) {
	    System.err.println("NQL ERROR ");
	    ex.printStackTrace();
	}
    }

    private void displayResults(List resultlist) {
	/* First result is the names of the parameters - ignore */
	for (int i=1; i<resultlist.size(); i++) {
	    System.out.print("Result " + i + ": ");
	    List ellist=(List)resultlist.get(i);
	    for (int j=0; j<ellist.size(); j++) {
		NOMElement ne=(NOMElement)ellist.get(j);
		System.out.print(ne.getName() + " " + ne.getID() + "; ");
	    }
	    System.out.print("\n");
	}
    }

    public static void main(String args[]){
	String corpus=null;
	String query=null;

	if (args.length < 2 || args.length > 4) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-query") || flag.equals("-q")) {
		i++; if (i>=args.length) { usage(); }
		query=args[i];
	    } else {
		usage();
	    }
	}
	if (corpus==null || query==null) { usage(); }
	
	SimpleQueryDisplay psv = new SimpleQueryDisplay (corpus, query);
    }

    private static void usage () {
	System.err.println("Usage: java SimpleQueryDisplay -corpus <path-to-metadata> -query <NQL-query> ]");
	System.exit(0);
    }

}
