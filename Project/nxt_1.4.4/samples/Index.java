import java.util.*;

import net.sourceforge.nite.search.*;
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;

import net.sourceforge.nite.nom.NOMException;

/**
 * Simple program that adds indices to an NXT-format corpus
 * using the query language.  
 * 
 * java Index -c CORPUS -o OBS -q QUERY -t TAG -r ROLE1 ROLE2 ... ROLEM
 *
 * where CORPUS is the location of a metadata file
 * 
 * OBS is optional, and is the name of an observation.  If it is omitted,
 * all observations named in the metadata file are indexed in turn.
 *
 * QUERY is a query.  Let n be the number of unquantified variables in 
 * the first subquery (unquantified meaning, without a forall or exists).
 *
 * TAG is the name of a tag (code).  It is optional and defaults to "markable".
 *
 * ROLE1 to ROLEM are optional, and are the names of roles to use in
 * the indexing.
 *
 * The -r flag, if present, must be given last.
 *
 * For each query match, a new tag of type TAG is added to the corpus.
 * If -r is omitted, the new tag is made a parent of the first unquantified
 * variable of the query.  If -r is included, then the new tag will contain
 * m pointers which point to the first m unquantified variables in the 
 * first subquery, using the given role names in order.  m must be less
 * than or equal to n.  
 *
 * The program assumes that the new  tags have already been
 * added into the metadata file (in a new coding, since no further
 * work is done to attach the markables to parents or anything else).
 * If the indexing adds parents, then the type of the coding file
 * (interaction or agent) must match the type of the coding file 
 * that contains the matches to the first variable.  
 * If an observation name is passed, it creates a index only for the one
 * observation; if none is, it indexes each observation in the metadata
 * file by loading one at a time (so this won't work for queries that 
 * need comparisons across observations).  
 *
 * Note that this does not restrict the same element to at most one match, 
 * even though that's a property we often want our mappings to have.  
 * When creating indices of one variable, it is often best to use only
 * one unquantified variable in the first subquery of the query, so that
 * we don't index the same thing more than once (i.e., using further subqueries
 * as a filter, not to retain further variable matches). Also note that Index
 * does not remove existing tags of the same type before operation - it just
 * adds new tags.  Making the same call twice will create two indices to the
 * same data, which is usually undesirable, but this enables an index to be
 * built up gradually using several different queries.
 *
 * 
 * The canonical metadata form for an index file, assuming roles are used, is
 * 
 *             <coding-file name="FOO">
 *               <featural-layer name="BAZ">
 *                   <code name="TAGNAME">
 *                   	<pointer number="1" role="ROLENAME" target="LAYER_CONTAINING_MATCHES"/>
 *                   </code>
 *               </featural-layer>
 *          </coding-file>
 * 
 *  
 * An earlier version of this program was called AddMarkables.
 * It only worked for one observation at a time
 * and didn't take the arguments as flags but in a particular order.
 * We've changed the name only because this isn't backward compatible;
 * the previous version didn't take the arguments as flags but in 
 * a particular order.
 * 
 *
 ********************************************************************
 * KNOWN PROBLEMS AND FUTURE DEVELOPMENT
 ********************************************************************
 * 
 * No known bugs.
 * 
 * Further versions might usefully 
 * 
 * (1) modify the metadata file (if the API allows this sort of change)
 * (2) warn if there are any markables before we start (since this will 
 * add new ones to the existing set)
 * (3) write the output in a different place than the input 
 * 
 * @author Jean Carletta, Oct 2003 - Feb 2005
 **/

public class Index {

	/*	static final String META = "meta";
		static final String CORPUSID = "corpusid";
		static final String CORPUSDESC = "corpusdesc";
		static final String CODING = "coding";
		static final String ONTOLOGY = "ontology";
		static final String OBJECTSET = "objectset";
		static final String EXPORTDIR = "exportdir";
		static final String XP = "xpointer";
		*/
	NOMWriteCorpus nom;
	NiteMetaData controlData;
	String corpusname;

	private Engine searchEngine = new Engine();

	public Index(String c, String o, String q, String t, List rolenames) {
	    corpusname = c;
	    try {
		controlData = new NiteMetaData(corpusname);
	    } catch (NiteMetaException nme) {
		nme.printStackTrace();
	    }
	    if (controlData.getElementByName(t) == null) {
		System.out.println(
				   "Metadata contains no such tag name; exiting...");
		System.exit(0);
	    }
	    if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
		    if (o == null) {
			// no observation name passed, do whole corpus one by one
			try {
			    // second arg sends log messages to System.err, not System.out.
			    nom = new NOMWriteCorpus(controlData, System.err);
			    // temporarily disable lazy loading until bug gets fixed
			    // nom.setLazyLoading(false);
			    //JC reinstating lazy loading 19 Mar 2010
			    List obslist = controlData.getObservations();
			    for (int i = 0; i < obslist.size(); ++i) {
				NiteObservation nobs =
				    (NiteObservation) obslist.get(i);
				o = nobs.getShortName();
				nom.loadData(nobs);
				addIndicesForObservation(nom, o, q, t, rolenames);
				try {
				    /* metadata shouldn't change in this operation */
				    /* controlData.writeMetaData(corpusname); */
				    //nom.serializeCorpus();
				    nom.serializeCorpusChanged();
				} catch (Exception e) {
				    e.printStackTrace();
				}
				nom.clearData();
			    }
			} catch (Exception e) {
			    e.printStackTrace();
			}
		    } else { //one observation only
			try {
			    NObservation obs =
				controlData.findObservationWithName(o);
			    nom = new NOMWriteCorpus(controlData, System.err);
			    //temporarily disable lazy loading until bug gets fixed
			    //nom.setLazyLoading(false);
			    //JC reinstating lazy loading 19 Mar 2010
			    nom.loadData(obs);
			    addIndicesForObservation(nom, o, q, t, rolenames);
			    try {
				/* metadata shouldn't change in this operation */
				/* controlData.writeMetaData(corpusname); */
				//nom.serializeCorpus();
				nom.serializeCorpusChanged();
			    } catch (Exception e) {
				e.printStackTrace();
			    }
			} catch (NOMException nex) {
			    nex.printStackTrace();
			}
		    }
		} else {
		    System.out.println(
				       "This is a simple (one document) corpus: exiting...");
		}
	}

	private void addIndicesForObservation(
		NOMWriteCorpus nom,
		String o,
		String q,
		String t,
		List rolenames) {
		List elist = null;
		try {
			elist = searchEngine.search(nom, q);
		} catch (Throwable e) {
			e.printStackTrace();
			System.exit(0);
		}
		Iterator elit = elist.iterator();
		/* The first thing on the list returned by the search engine is a duff entry
		 * containing the names of the variables for the remaining things on the list.
		 * Use the boolean to get past that entry.  
		 */
		boolean first = true;
		while (elit.hasNext()) {
			List reslist = (List) elit.next();
			if (first) {
				first = false;
				continue;
			}
			/* For each value of reslist, we need to create one new markable element.
			 */
			try {
			    /* get the coding for the markable element handy, since we'll need to
			     * test the type (agent or interaction).
			     */
			    NElement marktype = controlData.getElementByName(t);
			    NLayer marklayer = marktype.getLayer();
			    NCoding markcoding = (NCoding) marklayer.getContainer();
			    /* If there are no rolenames, so that we're creating a parent of the match
			     * for the first unquantified variable, then we need to know the name of
			     * the agent (or null if none) before creating the new element.  The parent
			     * markable had better be of the same type (agent or interaction) as the result.
			     */
			    if (rolenames == null) {
				//System.out.println("No rolenames - adding first match as child.");
				/* get match of the first unquantified variable */
				NOMWriteElement result = (NOMWriteElement) reslist.get(0);
				NOMWriteAnnotation mark = null;
				String agent = result.getAgentName();
				if ((agent != null) && (markcoding.getType() == NCoding.AGENT_CODING)) {
				    mark = new NOMWriteAnnotation(nom, t, o, agent);
				    mark.addToCorpus();
				} else if  ((agent == null) && (markcoding.getType() == NCoding.INTERACTION_CODING)) {
				    mark = new NOMWriteAnnotation(nom, t, o, (String)null);
				    mark.addToCorpus();
				}
				else {
				    System.out.println("ERROR: Parent and child aren't both interaction tags or both agent tags - basic type must match.");
				    System.exit(0);
				}
				/* make markable have the result as its child */
				mark.addChild(result);
			    } else { 
				/* we're adding pointers with a set of roles.  First add the new element.
				 * We need to know whether it goes in an agent coding first, and then
				 * if there are multiple roles, each had better point to something by the
				 * same agent.
				 */
				//System.out.println(rolenames.size() + " rolenames.");
				String agent = null;
				NOMWriteAnnotation mark = null;
				if (markcoding.getType() == NCoding.AGENT_CODING) {
				    NOMWriteElement result = (NOMWriteElement) reslist.get(0);
				    agent = result.getAgentName();
				    mark = new NOMWriteAnnotation(nom, t, o, agent);
				} else {
				    mark = new NOMWriteAnnotation(nom, t, o, (String)null);
				}
				mark.addToCorpus();
				/* Iterate through the the matches and the rolenames at the same time, 
				 * adding pointers for them each.  
				 */
				Iterator roles_it = rolenames.iterator();
				Iterator results_it = reslist.iterator();
				while (roles_it.hasNext() && results_it.hasNext()) {
				    String rolename = (String) roles_it.next();
				    NOMWriteElement result = (NOMWriteElement) results_it.next();
				    if (agent != null) {
					/* check that this result is from the same agent as the markable */
					String result_agent = result.getAgentName();
					if (!(result_agent.equals(agent))) {
					    System.out.println("WARNING:  adding a tag in an agent coding but adding roles that point to matches from more than one agent.");
					}
				    }
				    /* make the result be pointed to by the new markable. */
				    //System.out.println("Adding pointer with role " + rolename);
				    mark.addPointer(new NOMWritePointer(nom, rolename, mark, result));
				}
				/* We know that the rolename list is supposed to be the shorter of the two lists
				 * if there's a length difference; complain if not. */

				if (roles_it.hasNext()) {
				    System.out.println("WARNING:  found more rolenames than unquantified variables.");
				}
			    }
			} catch (NOMException nom_e) {
				nom_e.printStackTrace();
				System.exit(0);
			}

		}
	}

	/**
	* Called to start the  application.
	* Legal command line arguments are:
	*<ul>
	*<li> -c corpus </li>
	* <li> -q query </li>
	* <li> -o observation </li>
	* <li> -t tagname </li>
	* <li> -r rolename1 rolename2 ... rolenamem </li>
	*</ul>
	*
	*/
	public static void main(String args[]) {
		String corpus_name = null;
		String query = null;
		String obs_name = null;
		String tag_name = "markable";
		List role_names = null;
		if (args.length < 4) {
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
				obs_name = args[i];
			} else if (flag.equals("-tag") || flag.equals("-t")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				tag_name = args[i];
			} else if (flag.equals("-role") || flag.equals("-r")) {
				i++;
				if (i >= args.length) {
					usage();
				}
				role_names = new ArrayList();
				for (int j=i;j<args.length;++j) {
				    role_names.add(args[j]);
				}
				i = args.length;
			} else {
				usage();
			}
		}
		if ((corpus_name == null) || (query == null)) {
			usage();
		}

		Index m = new Index(corpus_name, obs_name, query, tag_name, role_names);
	}

	private static void usage() {
		System.err.println(
			"Usage: java Index -corpus <path-to-metadata> -observation <observation-name> -query <NQL-query> -tag <tag-name> -role <role-name1> <role-name2> <role-namem>");
		System.exit(0);
	}

}
