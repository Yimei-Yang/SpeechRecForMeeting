import java.io.*;

import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.nom.NOMException;

/**
 * A simple demonstrator of the NOM and metadata libraries.
 * Call like this:
 *    java SimpleSaveExample -corpus Data/meta/mock.xml
 * @author Jonathan Kilgour, Jan 2003
 **/

public class SimpleSaveExample { 
    
    NOMWriteCorpus nom;
    NiteMetaData meta;
    String corpusname;

    public SimpleSaveExample(String corp) {
	corpusname = corp;

	try {
	    /* First load the metadata */
	    meta = new NiteMetaData(corpusname);
	    
	    /* set up a new NOM Corpus */
	    nom = new NOMWriteCorpus(meta);

	    /* set lazy loading off so that all data is loaded whether
	     * it's used or not */
	    nom.setLazyLoading(false);
	    
	    /* Load all the data in the corpus 
	       (according to the metadata) */
	    nom.loadData();
		
	    /* Set the directories to save data */
	    meta.setCodingPath("Newcorpus"); // relative to metadata!
	    meta.setOntologyPath("Newcorpus");
	    
	    /* Set the link style to XLink / XPointer style */
	    meta.setLinkType(NMetaData.XPOINTER_LINKS);
	    
	    /* Write the changed metadata */
	    meta.writeMetaData("New/newmetadata.corpus");
		
	    /* Write the corpus data to the rel. dirs specified above*/
	    nom.serializeCorpus();		
		
	} catch (NOMException nex) {
	    nex.printStackTrace();
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}

    }


    /**
     * Called to start the  application.
     */
    public static void main(String args[]){
	if (args.length != 2) { usage(); }
	String flag = args[0];
	String corpusn = args[1];
	if (! flag.equals("-corpus")) { usage(); }
	if (corpusn == null) { usage(); }
	SimpleSaveExample m = new SimpleSaveExample(corpusn);
    }

    private static void usage () {
	System.err.println("Usage: java SimpleSaveExample -corpus <path-to-metadata>");
	System.exit(0);
    }
}

