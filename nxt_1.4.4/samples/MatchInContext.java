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
 * Simple program that runs a query on 
 * a corpus and shows the matches in some surrounding
 * context.  It takes 
 * the following arguments:
 * 
 * -corpus CORPUS
 * -query QUERY
 * -context CONTEXT
 * -textatt TEXTATT
 * -observation OBS
 * -surround
 * -allatonce
 * 
 * where CORPUS is the location of the metadata file, 
 * 
 * QUERY is a query expressed in NQL that one wishes to match; results
 * will be shown for the first variable given in the query.
 * 
 * CONTEXT is an optional query expressed in NQL that expresses the
 * surrounding context one wishes to show the matches in but does
 * not express a relationship to the main query --- we add that the
 * first variable in the context query dominates the first variable
 * in the main query. If CONTEXT is not given, no context is shown.
 * The context query must not share variable names with the main query.
 * 
 * TEXTATT is optional and indicates the name of the attribute
 * where the orthography is stored (for when the orthography isn't
 * stored as PCDATA as in the TEI convention).  Note that this option
 * is also useful for returning a bare list of attribute values for
 * further processing, if CONTEXT is not given.
 * 
 * OBS is the name of a single observation from the corpus to run over.
 * 
 * SURROUND means show the preceding and following siblings of whatever
 * the context matches as well.  These are siblings in file order, and
 * may or may not correspond to anything useful. Defaults to no surround.
 *
 * ALLATONCE means load all observations at once, which is necessary 
 * when the query has conditions over multiple observations.  Defaults
 * to not allatonce, which is faster; the default is the same as the
 * old "onebyone" behaviour, which previously was a different program.
 *
 * The text is shown on STDOUT, with the query match upcased within the
 * context if context is specified. 
 * 
 * There is no clean way of knowing where to insert line breaks,
 * speaker attributions, etc. in a general utility such as this 
 * one; for better displays write a tailored interface.
 * 
 * Canonical usage would use one variable queries for the main and
 * context queries, with the latter expressing simply a type, possibly 
 * with some constraints on attribute values.
 * 
 * There may be less or more than one context match for a query result;
 * in these cases we comment and show the first match we find (if any).
 * 
 * The display ends up on System.out; everything else that could
 * potentially be on System.out is redirected to System.err.
 * 
 * 
 * @author Jean Carletta, Aug 2003
 **/

public class MatchInContext {

	NOMWriteCorpus nom;
	NiteMetaData controlData;
	String corpusname;

	private Engine searchEngine = new Engine();

	public MatchInContext(String c, String q, String con, String t, String o, boolean s, boolean allatonce) {
		corpusname = c;
		try {
			controlData = new NiteMetaData(corpusname);
		} catch (NiteMetaException nme) {
			nme.printStackTrace();
		}
		if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
			try {
				// second arg sends log messages to System.err, not System.out.
				nom = new NOMWriteCorpus(controlData, System.err);
				if (o != null) {
					NObservation obs = controlData.findObservationWithName(o);
					nom.loadData(obs);
					ShowMatchText(nom, q, t, con, obs.getShortName(),s);
				} else
				    if (allatonce) {
					nom.loadData();
					ShowMatchText(nom, q, t, con, null,s);
				    } else {
					List obslist = controlData.getObservations();
					for (int i=0;i<obslist.size() ;++i) {
					    NiteObservation nobs = (NiteObservation) obslist.get(i);
					    nom.loadData(nobs);
					    ShowMatchText(nom, q, t, con, nobs.getShortName(),s);
					    nom.clearData();
					};
				    };
			} catch (NOMException nex) {
			    nex.printStackTrace();
			}
		} else {
			System.err.println(
				"This is a simple (one document) corpus: exiting...");
		}

	}

	/*  
	 */
	void ShowMatchText(NOMWriteCorpus nom, String q, String t, String con, String dial, boolean surround) {
		List elist = null;
		try {
			elist = searchEngine.search(nom, q);
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(0);
		}
		/* The first thing on the list returned by the search engine is a duff entry
		 * containing the names of the variables for the remaining things on the list.
		 * I'm assuming we just want to show the first variable given, so that will
		 * be the first thing in every other entry.
		 */
		for (int i = 1; i < elist.size(); ++i) {
		    //System.out.print("RESULT " + i + ", in " + dial + " ");
			List resultlist = (List) elist.get(i);
			NOMElement ne = (NOMElement) resultlist.get(0);
			//TEMP - this here should change so that it figures out the observation, 
			// not gets passed it.
			System.out.print("RESULT " + i + ", observation " + dial + ", id " + ne.getID() + " ");

			if (con == null) {
				//if we don't want to show in context, 
				//treewalk children and grab orthography 
				System.out.println(":");
				ShowMatchElement(ne, t, false, null);
			} else {
				//find the ancestor of the given type and from there,
				//treewalk children, upcasing when we get to the given id.
				/// to find ancestor, make a complex query where the first
				/// part specifies the thing with the id of the match element
				/// and the second specifies the query given at command line
				/// with the added conjunct that it dominates the match from
				/// the first query.   
				int condollar = con.indexOf('$');
				int conspace = con.indexOf(' ', condollar);
				String conmatchvar = null;
				if ((condollar > -1) && (conspace > 0)) {
					conmatchvar = con.substring(condollar, conspace);
				};
				int qdollar = q.indexOf('$');
				int qspace = q.indexOf(' ', qdollar);
				String qmatchvar = null;
				if ((qdollar > -1) && (qspace > 0)) {
					qmatchvar = q.substring(qdollar, qspace);
				};
				if ((conmatchvar != null) && (qmatchvar != null)){
                                    // if context query ends in :, with no conditions, leave
				    // out the && to join up.
				    String joinstring = " && (";
				    if (con.trim().lastIndexOf(':') == con.trim().length() - 1) {
                                        joinstring = " (";
				    };
					String complexq =
						"(" + qmatchvar 
							+ " " + ne.getName()
							+ "):(ID(" + qmatchvar 
							+ ") == '"
							+ ne.getID()
							+ "')::"
							+ con
							+ joinstring
							+ conmatchvar
							+ " ^ " 
							+ qmatchvar 
							+ ")";
					//System.out.println(complexq);
					List complexelist = null;
					try {
						complexelist = searchEngine.search(nom, complexq);
					} catch (Throwable e) {
						e.printStackTrace();
						System.exit(0);
					};
					// it's possible there is no context match, or more than one;
					// use first one if possible but warn, and if none, just show bare, but
					// upcased
					if (complexelist.size() < 2) {
						// no results means no context matches
						System.out.println("(no matching context):");
						ShowMatchElement(ne, t, true, null);
					} else {
						// need to find match to context query var, which is one level
						// down.
						// first, get first match to complete query.  This is a hierarchical
						// list with, at top level, the variable from the main query, followed
						// by a list of matches to the context query var.
						List contextvarmatchlist = (List) complexelist.get(1);
						if (contextvarmatchlist.size() > 2) {
							System.out.println("(first of multiple context matches):");
						} else {
							System.out.println(":");
						};
						///System.out.println(contextvarmatchlist);
						// instead of starting to show at ne, start at the match
						// to the context element, which can be found in the result
						// list, not at the top level, but one level down hierarchically.
						List complexresultlist =
							(List) contextvarmatchlist.get(1);
						List containscontextmatch =
							(List) complexresultlist.get(1);
						NOMElement cne =
						    (NOMElement) containscontextmatch.get(0);
                                                if (surround) {
						    NOMElement prev = cne.getPreviousSibling();
						    if (prev != null) {
							System.out.print("(previous) ");
							ShowMatchElement(prev, t, false, null);
							System.out.println("");
						    };
						};
						ShowMatchElement(cne, t, false, ne.getID());
                                                if (surround) {
						    NOMElement next = cne.getNextSibling();
						    if (next != null) {
							System.out.println("");
							System.out.print("(next) ");
							ShowMatchElement(next, t, false, null);
							System.out.println("");
						    };
						};
					}
				} else { // couldn't find the matchvar; treat as no matching context
					System.out.println(
						"(no matching context because couldn't parse context query):");
					ShowMatchElement(ne, t, false, null);
				}
			};

			System.out.print("\n\n");
		}
	}

	/**Show any orthography associated with this element
	  * and then recurse to children.  The orthography
	  * should really only be in a flat list at the bottom
	  * level, but we look for it everywhere in case people want
	  * to misuse the system by asking for some non-orthography
	  * attribute like ids (but in this case, they have to live
	  * with this traversal order and no indication of the hierarchy).
	  * When we hit the element with matchid, we display it (and its
	  * children) in upper case; everything else we deliberately
	  * downcase.
	  */
	void ShowMatchElement(
		NOMElement ne,
		String t,
		boolean upcase,
		String matchid) {
		// next line temporary
		//System.out.print("ID" + ne.getID() + "ID ");
		String orth = null;
		if (t == null) {
			orth = ne.getText();
		} else {
			orth = (String) ne.getAttributeComparableValue(t);
		}
		if (ne.getID() == matchid) {
			upcase = true;
		};
		if (orth != null) {
		    if (upcase == true) {
			orth = orth.toUpperCase(); 
		    } else {
			orth = orth.toLowerCase(); 
		    };
		    System.out.print(orth + " ");
		};
		List childlist = ne.getChildren();
		if (childlist != null) {
		    for (int i = 0; i < childlist.size(); ++i) {
			ShowMatchElement(
					 (NOMElement) childlist.get(i),
					 t,
					 upcase,
					 matchid);
		    }
		} else {
		    // it's confusing when the match has no orthography;
		    // in these cases stick in a minimal visual trace.
		    if ((orth == null) && (upcase == true)) {
			System.out.print("[NULL] ");
		    }
		}
	}

	/**
	 * Called to start the  application.
	 * Legal command line arguments are:
	 *<ul>
	 * -corpus CORPUS
	 * -observation OBS
	 * -query QUERY
	 * -context CONTEXT
	 * -textatt TEXTATT
	 */

	public static void main(String args[]) {
		String corpus_name = null;
		String query = null;
		String observation_name = null;
		String context = null;
		String textatt = null;
		boolean surround = false;
		boolean allatonce = false;
		if (args.length < 4 || args.length > 12) {
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
			} else if (flag.equals("-query") || flag.equals("-q")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				query = args[i];
			} else if (flag.equals("-observation") || flag.equals("-o")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				observation_name = args[i];
			} else if (flag.equals("-context")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				context = args[i];
			} else if (flag.equals("-textatt") || flag.equals("-t")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				textatt = args[i];
			} else if (flag.equals("-surround") || flag.equals("-s")) {
                            surround = true;
			} else if (flag.equals("-allatonce") || flag.equals("-a")) {
                            allatonce = true;
			} else {
				usage();
			}
		}
		if ((corpus_name == null) || (query == null)) {
			usage();
		}
		MatchInContext m =
			new MatchInContext(
				corpus_name,
				query,
				context,
				textatt,
				observation_name,
				surround,
				allatonce);
		
	}


	private static void usage() {
		System.err.println(
			"Usage: java MatchInContext -corpus <path-to-metadata> -query <NQL-query> -context <NQL-query> -textatt <ATTNAME> -observation <OBSNAME> -surround -allatonce");
		System.exit(0);
	}

}
