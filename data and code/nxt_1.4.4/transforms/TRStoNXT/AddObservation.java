import java.util.*;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;

import net.sourceforge.nite.nom.NOMException;

/**
 * Simple program that adds an observation to the metadata file, 
 * if it isn't already named in there.  Two arguments at the command
 * line; the path to the metadata file and the observation name.
 * 
 * @author Jonathan Kilgour, Jean Carletta April 2003
 **/

public class AddObservation {

    NOMWriteCorpus nom;
    NiteMetaData meta;
    String corpusname;
    
    public AddObservation(String c, String o) {
	corpusname = c;
	if (o == null) {
		System.err.println("Passed null observation name -- doing nothing!");
		System.exit(0);
	};
	try {
	    meta = new NiteMetaData(corpusname);
	    NiteObservation ob = (NiteObservation) meta.getObservationWithName(o);
	    if (ob==null) {
		System.err.println("Adding observation with name " + o + "...");
		ob = new NiteObservation(meta, o, "");
		meta.addObservation(ob);
		meta.writeMetaData(c);
	    } else {
		System.err.println("Observation with name " + o + " already present; loading");
	    }
	    nom = new NOMWriteCorpus(meta);
	    try {
		nom.setLazyLoading(false);
		nom.loadData(ob);
		nom.serializeCorpus();
	    } catch (NOMException nex) {
		nex.printStackTrace();
	    }

	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	    System.exit(0);
	}
    }

    /**
     * Called to start the  application.
     * Legal command line arguments are:
     *<ul>
     *<li> corpus </li>
     *<li> observation </li>
     *</ul>
     *
     * Does nothing if the observation already exists.
     * */
    public static void main(String args[]) {
	if (args.length != 2) {
	    usage();
	}
	String corpus_name = args[0];
	String observation_name = args[1];
	AddObservation m = new AddObservation(corpus_name, observation_name);

    }

    private static void usage() {
	System.err.println("Usage: java AddObservation <path-to-metadata> <observation-name>");
	System.exit(0);
    }

}
