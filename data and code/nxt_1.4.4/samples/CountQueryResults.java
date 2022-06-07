import java.util.*;
//import java.io.FileOutputStream;
//import java.io.PrintStream;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;

import net.sourceforge.nite.nom.NOMException;

/**
 * Simple program that runs a query on a loaded observation
 * from a particular corpus and counts the query results. It takes 
 * the following arguments:
 * 
 * -corpus CORPUS
 * -query QUERY
 * -observation OBS
 * -allatonce
 * 
 * where CORPUS is the location of the metadata file.
 * 
 * QUERY is a query expressed in NQL.  Note that in the case of
 * queries with multiple variables, the count will be of the number
 * of matching n-tuples, not of the number of matches to the first
 * named variable.  Also remember that in complex queries, any matches
 * to a query for which there are no matches to the subsequent query
 * will be removed from the result list.
 * 
 * OBS is the name of a single observation from the corpus to run over.
 * 
 * ALLATONCE means load all observations at once, which is necessary 
 * when the query has conditions over multiple observations.  Defaults
 * to not allatonce, which is faster; the default is the same as the
 * old "onebyone" behaviour, which previously was a different program.
 * 
 * -allatonce is incompatible with -observation OBS; the latter overrides
 * the former if both are specified.
 * 
 * The count ends up on System.out; everything else that could
 * potentially be on System.out is redirected to System.err.  Where
 * -observation OBS or -allatonce is used, the bare count is given.
 * Otherwise, the output is one line per observation, with the short
 * name of the observation (as in the metadata file) and the count
 * for that observation separated by whitespace.
 * 
 * @author Jean Carletta, Aug 2003
 **/

public class CountQueryResults {

	NOMWriteCorpus nom;
	NiteMetaData controlData;
	String corpusname;

	private Engine searchEngine = new Engine();

	public CountQueryResults(String c, String o, String q, boolean allatonce) {
		corpusname = c;
		try {
			controlData = new NiteMetaData(corpusname);
		} catch (NiteMetaException nme) {
			nme.printStackTrace();
		}
		if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
			// //Commented-out version separates log and error messages; putting the
			// // log stuff to a file; it works.
			//try {
			//	FileOutputStream fileout = new FileOutputStream("logfile");
			//	PrintStream msgs = new PrintStream(fileout);

			try {
				// second arg sends log messages to System.err, not System.out.
				nom = new NOMWriteCorpus(controlData, System.err);
				//nom = new NOMWriteCorpus(controlData, msgs);
				//nom.setErrorStream(msgs);
				if (o != null) {
					NObservation obs = controlData.findObservationWithName(o);
					nom.loadData(obs);
					CountMatches(nom, q, null);
				} else if (allatonce) {
					nom.loadData();
					CountMatches(nom, q, null);
				} else {
					List obslist = controlData.getObservations();
					for (int i = 0; i < obslist.size(); ++i) {
						NiteObservation nobs = (NiteObservation) obslist.get(i);
						nom.loadData(nobs);
						CountMatches(nom, q, nobs.getShortName());
						nom.clearData();
					};
				};
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

	/* obsname is null when counting over whole corpus 
	 */
	void CountMatches(NOMWriteCorpus nom, String q, String obsname) {
		List elist = null;
		try {
			elist = searchEngine.search(nom, q);
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(0);
		}
		if (obsname != null) {
			System.out.print(obsname + "   ");
     	}	
		/* The first thing on the list returned by the search engine is a duff entry
		 * containing the names of the variables for the remaining things on the list.
		 * Return length of list, minus this one.  But first, check the size - if it's zero, 
		 * that means no matches and we want to return 0, not -1!
		 */
		if (elist.size() == 0) {
			System.out.println(0);
		} else {
			System.out.println(elist.size() - 1);
		}
	}

	/**
	 * Called to start the  application.
	 * Legal command line arguments are:
	 *<ul>
	 *<li> -c corpus </li>
	 * <li> -o observation </li>
	 * <li> -q query </li>
	 *</ul>
	 *
	 */
	public static void main(String args[]) {
		String corpus_name = null;
		String observation_name = null;
		String query = null;
		boolean allatonce = false;
		if (args.length < 2 || args.length > 7) {
			usage();
		}
		for (int i = 0; i < args.length; i++) {
			String flag = args[i];
			if (flag.equals("-corpus") || flag.equals("-c")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				corpus_name = args[i];
			} else if (flag.equals("-observation") || flag.equals("-o")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				observation_name = args[i];
			} else if (flag.equals("-query") || flag.equals("-q")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				query = args[i];
			} else if (flag.equals("-allatonce") || flag.equals("-a")) {
			    allatonce = true;
			} else {
				usage();
			}
		}
		if ((corpus_name == null) || (query == null)) {
			usage();
		}

		CountQueryResults m =
			new CountQueryResults(corpus_name, observation_name, query, allatonce);
	}

	private static void usage() {
		System.err.println(
			"Usage: java CountQueryResults -corpus <path-to-metadata> -observation <observation-name> -query <NQL-query> -allatonce");
		System.exit(0);
	}

}
