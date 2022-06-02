import java.util.*;
import java.io.*;
//import java.io.FileOutputStream;
//import java.io.PrintStream;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;

import net.sourceforge.nite.nom.NOMException;

/**
 * MAJOR CHANGE IN VERSION 1.4:  -INDEPENDENT IS DEFAULT BEHAVIOUR,
 * USE -ALLATONCE OTHERWISE.
 * 
 * Simple program that runs a query on a loaded observation
 * from a particular corpus and saves the query result XML. It 
 * takes the following arguments:
 * 
 * -corpus CORPUS
 * -observation OBS
 * -query QUERY
 * -independent or -allatonce
 * -filename OUTFILENAME 
 * -directory DIRNAME
 * [ -nolazy ] 
 * 
 * where CORPUS is the location of the metadata file, OBS is an
 * observation name (if not given, works over all observations listed in
 * the metadata file), and QUERY is a query expressed in NQL. 
 * If -allatonce is indicated, then it loads the entire corpus in
 * one go and saves one result file.  If -independent is indicated,
 * then it loads one observation at a time and saves one result file
 * per observation.  The default is -independent. 
 * 
 * If no filename is indicated, the output goes to System.out.
 * (Note that this isn't very sensible to do in conjunction with
 * -independent because the output will just concatenate separate
 * XML documents.)
 * Everything else that could potentially be on System.out is 
 * redirected to System.err.  Example usage:  
 * java SaveQueryResults -c CORPUS -q QUERY > out >& err
 * or
 * java SaveQueryResults -c CORPUS -q QUERY 2> err | less
 * 
 * If a filename is indicated,
 * the output ends up in the directory named by DIRNAME.
 * The default is the current directory.
 * It ends up in OUTFILENAME unless -independent is indicated,
 * in which case that filename is prefixed with the name of the 
 * observation and a full stop (.).  
 * -independent is ignored if -observation OBS is indicated (i.e., 
 * the output is saved without prefixing the filename).
 *
 * if the -nolazy flag is set lazy loading is turned off so all files
 * are loaded for the observation (mainly useful for debugging)
 * 
 * When I run this utility under cygwin, I have to use Windows-style
 * directory naming to save where I expect; e.g., -d "C:", not 
 * -d "/cygdrive/c".  I suspect this is configurable but I'm not
 * sure how to change it.  This aspect of the program has not been
 * tested on other platforms.
 * 
 * @author Jean Carletta, Dec 2003
 **/

public class SaveQueryResults {

	NOMWriteCorpus nom;
	NiteMetaData controlData;
	String corpusname;

	private Engine searchEngine = new Engine();

	public SaveQueryResults(String c, String o, String q, String f, String d, boolean ind, boolean eager) {
		corpusname = c;
		try {
			controlData = new NiteMetaData(corpusname);
		} catch (NiteMetaException nme) {
			nme.printStackTrace();
		}
			if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
				//alt Commented-out version separates log and error messages; putting the
				//alt log stuff to a file; it works.
			    //alt try {
			    //alt	FileOutputStream fileout = new FileOutputStream("logfile");
				//alt	PrintStream msgs = new PrintStream(fileout);
				// figure out the directory where the output will go, if one
				// hasn't been specified
				if (d == null) {
				   File dir = new File (".");
                   try {
                      d = dir.getCanonicalPath();
       			   }
     				catch(Exception e) {
       					e.printStackTrace();
       				}
     			}	
				try {
					// second arg sends log messages to System.err, not System.out.
					nom = new NOMWriteCorpus(controlData, System.err);
					if (eager) {
					    nom.setLazyLoading(false);
					}
					//alt nom = new NOMWriteCorpus(controlData, msgs);
					//alt nom.setErrorStream(msgs);
					// if we're dealing with a whole corpus at once, 
					// load the lot and run the query
					if ((o == null) && (ind == false)) {
						nom.loadData();
						RunQueryAndSaveResults(nom, q, f, d);
					} else {
				      // if we're dealing with one observation, just load it
				      // and run the query
					  if (o != null) {
					  NObservation obs = controlData.findObservationWithName(o);
					  nom.loadData(obs);
					  RunQueryAndSaveResults(nom, q, f, d);
					} else { 
						// independent is true; we need to loop over the observations
					    List obslist = controlData.getObservations();
                        for (int i=0;i<obslist.size() ;++i) {
                            NiteObservation nobs = (NiteObservation) obslist.get(i);
         				    nom.loadData(nobs);
         				    String file = null;

							
         				    if (f != null) {
         				    	file = nobs.getShortName() + "." + f;
         				    }
       
         				    RunQueryAndSaveResults(nom, q, file, d);
					        nom.clearData();
                        }	     
					  }
					}
				} catch (NOMException nex) {
					nex.printStackTrace();
				}
				//} catch (java.io.FileNotFoundException nex) {
				//	nex.printStackTrace();
				//}
			} else {
				System.err.println(
					"This is a simple (one document) corpus: exiting...");
			}

	}

	/*  
	 */
	void RunQueryAndSaveResults(
		NOMWriteCorpus nom,
		String query,
		String outputfilename,
		String directoryname) {
		List elist = null;
		try {
			elist = searchEngine.search(nom, query);
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(0);
		}
		// resultToXML just gives a String, not a DOM or similar.
		String result_string = searchEngine.resultToXML(elist);
		if (outputfilename != null) {
   		   try {
         FileOutputStream fout= new FileOutputStream(directoryname + File.separator + 
		 				    outputfilename);
	       BufferedOutputStream bout= new BufferedOutputStream(fout);
	       OutputStreamWriter out = new OutputStreamWriter(bout, "8859_1");
	       out.write(result_string + "\n");
	       out.flush();
	       out.close();
	       } catch (Exception e) {
	          e.printStackTrace();
		   }
		} else {
			System.out.println(result_string);
		}
		}	

	/**
	 * Called to start the  application.
	 * Legal command line arguments are:
	 *<ul>
	 *<li> -corpus corpus </li>
	 * <li> -observation observation </li>
	 * <li> -query query </li>
	 * <li> -filename outfilename </li>
	 * <li> -directory dirname </li>
	 * <li> -independent </li>
	 *</ul>
	 *
	 */
	public static void main(String args[]) {
	String corpus_name = null;
	String observation_name = null;
	String query = null;
	String filename = null;
	String directory = null;
	boolean independent = true;
	boolean found_dependence_flag = false;
	boolean nolazy = false;
	if (args.length < 2 || args.length > 12) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus_name = args[i];
	    } else if (flag.equals("-observation") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		observation_name=args[i];
	    } else if (flag.equals("-query") || flag.equals("-q")) {
		i++; if (i>=args.length) { usage(); }
		query=args[i];
	    } else if (flag.equals("-filename") || flag.equals("-f")) {
		i++; if (i>=args.length) { usage(); }
		filename=args[i];
	    } else if (flag.equals("-directory") || flag.equals("-d")) {
		i++; if (i>=args.length) { usage(); }
		directory=args[i];
	    } else if (flag.equals("-allatonce") || flag.equals("-a")) {
		if (found_dependence_flag) { usage(); }
		found_dependence_flag = true;
		independent=false;
	    } else if (flag.equals("-independent") || flag.equals("-i")) {
		if (found_dependence_flag) { usage(); }
		found_dependence_flag = true;
		independent=true;
	    } else if (flag.equals("-nolazy") || flag.equals("-n")) {
		nolazy = true;
	    } else {
		usage();
	    }
	}
		if ((corpus_name == null)
			|| (query == null)) {
			usage();
		}

		SaveQueryResults m =
			new SaveQueryResults(
				corpus_name,
				observation_name,
				query, 
				filename,
				directory,
				independent,
				nolazy);
	}

	private static void usage() {
		System.err.println(
			"Usage: java SaveQueryResults -corpus <path-to-metadata> -observation <observation-name> -query <NQL-query> -filename <OUTFILENAME> -directory <DIRNAME> (-independent OR -allatonce) (-nolazy)");
		System.exit(0);
	}

}
