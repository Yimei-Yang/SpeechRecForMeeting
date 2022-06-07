import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.search.*;
import net.sourceforge.nite.util.SearchResultTimeComparator;
import net.sourceforge.nite.nom.NOMException;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.Hashtable;
import java.util.Iterator;


/**
 *
 * NGramCalc is a utility for calculating n-grams from NXT format data.
 * The command line options for it are a bit complicated because it's
 * very flexible.  It can take all elements of a particular type or
 * all those resulting from the first matched variable of a query, as
 * long as they are timed, and put them in order of start time.  Then,
 * using as the states either the value of a named attribute, or the 
 * names of elements at the end of a named role, or the value of a 
 * named attribute on those elements, it will report n-gram frequencies -
 * but it must be possible from the metadata to get the complete list of
 * states for the utility to work.  It looks for the list as the 
 * enumerated attribute values of the appropriate tag or as the set of
 * codes allowed in the layer a role is declared to point to.  Some
 * NXT users effectively have enumerated attribute values but declare
 * them as strings - modifying the declaration to make it explicit will
 * enable the utility.
 * 
 * To call the utility, which is in builds after NXT 1.3.1 but otherwise
 * available from CVS, set the classpath in the usual way, and then use 
 * 
 * java NGramCalc -corpus metadata_file_name -observation observation_name -tag tagname -query query -att attname -role rolename -n positive_integer
 * 
 * -corpus is required; it is the path to the metadata file.
 * 
 * -observation is optional; if it is used, the n-grams are calculated
 * over one observation only, and if it is omitted, over all observations
 * listed in the metadata.  Although only one set of numbers is reported,
 * NXT loads only one observation at a time when calculating them.
 * 
 * -tag is required; it names the tag to use in finding the state names.
 * 
 * -query is optional; if given, then the program uses matches to the
 * first named variable as the elements from which to derive states.  If
 * it is not given, then the query is assumed to match all tags of the
 * type named using -tag.  Note that if a query is used, it is possible
 * to have the first named variable use a disjunctive type, but only if
 * the method for deriving states from the elements works for both types
 * and results in the same enumerated list.  In this case, either of them
 * can be named in -tag.
 * 
 * -role is optional; if given, rather than looking for the states on the
 * query matches (or named tag if no query was given), the program looks
 * for them on the element found by tracing the named role from there.
 * This level of indirection is useful if the data was produced using one
 * of NXT's configurable end user tools, which tend to point to external
 * corpus resources to get possible annotation values.
 * 
 * -att; if given, uses the value of the named attribute both for finding
 * the possible state names and for finding the actual states.  -att is
 * required if -role is omitted, but optional if it is included.  If
 * -role is included and -att is omitted, then instead of using attribute
 * values, the states are derived from the element names in the layer
 * pointed to by the named role.
 * 
 * -n is optional; it gives the size of the n-grams.  It defaults to 1.
 * 
 * -independent is optional; if given, then the utility will only count
 * n-grams that do not overlap in items, rather than all n-grams.  
 * Traditionally, ABCDEF would have 4 tri-grams in it, but we have one 
 * user who wants just 2 tri-grams out of it, ABC and DEF.  She can get
 * this if she uses the -i option.  -independent defaults to false.
 * 
 * For instance,
 * 
 * java NGramCalc -c METADATA -t turn -a fs -n 3 
 * 
 * will calculate trigrams of fs attributes of turns and output a 
 * tab-delimited table like
 * 
 * 500	newfloor	floor	broken
 * 0	newfloor	newfloor	newfloor
 * 
 * Suppose that the way that the data is set up includes an additional
 * attribute value that we wish to skip over when calculating the tri-grams,
 * called "continued".
 * 
 * java NGramCalc -c METADATA -t turn -a fs -n 3 -q '($t turn):($t@fs != "continued")'
 * 
 * will do this. Entries for "continued" will still occur in the output
 * table because it is a declared value, but will have zero in the
 * entries. 
 * 
 * java NGramCalc -c METADATA -t gesture-target -a name -n 3 -q '($g gest):' -r gest-target
 * 
 * will produce trigrams where the states are found by tracing the gest-target
 * role from gest elements, which finds gesture-target elements (canonically,
 * part of some corpus resource), and further looking at the values of their
 * name attributes.  Note that in this case, the type given in -t is what results
 * from tracing the role from the query results, not the type returned in the
 * query.
 * 
 * java NGramCalc -c METADATA -t gest -q '($g gest):' -r gest-target
 * 
 * will produce unigrams where the states are named in the elements reached
 * by tracing the gest-target role from gest elements.  Again, canonically
 * these would be part of some corpus resource, but in this case the element
 * names themselves are used.  Note that in this case, type given in -t and
 * in the query results are the same.
 * 
 * At 21 Feb 05, use of -role without -att is not yet implemented.
 * 
 * We can think of the following changes that could be useful:
 * 
 *   allow order to be derived from end time rather than start time
 * 
 *   allow order to be derived from a structural order,
 *   not just a temporal one, so that the utility could be used on untimed
 *   data  
 * 
 * BUG AT 02 MAR 05:  people might expect to be able to pass in the agent
 * reserved attribute, but it doesn't work (NiteElement.getAttributeByName
 * doesn't find them, and at any rate, it's not of type enumerated in the
 * metadata).  Asking for agent is reasonable in this program.
 *
 * @author Jean Carletta, Jonathan Kilgour Feb 2005
 **/

public class NGramCalc { 
    
    // the data 
    private NOMWriteCorpus nom;
    private NiteMetaData controlData;

    // the arguments passed in
    String corpusname;
    String observationname;
    String query;
    String tagname;
    String attname;
    String rolename;
    String number;
    boolean independent; 
    private Engine searchEngine = new Engine();
    private Comparator mycomp = new SearchResultTimeComparator();

    public NGramCalc (String c, String o, String t, String q, String a, String r, String num, boolean independent) {

	corpusname = c;
	observationname = o;
	tagname = t;
	query = q;
	attname = a;
	rolename = r;
	number = num; 

	try {
	    controlData = new NiteMetaData(corpusname);
	} catch (NiteMetaException nme) {
	    nme.printStackTrace();
	}
	if (controlData.getCorpusType() == NMetaData.STANDOFF_CORPUS) {
	    /* need to find out what the possible attribute values are */
	    /* then need to count them them into an array */
	    NElement tag = controlData.getElementByName(tagname);
	    NLayer layer = tag.getLayer();
	    // Might as well check that layer is timed or can 
	    // inherit times here, although that's no guarantee
	    // timings actually exist, and can't check when we're
	    // going via a role because the named tag might be 
	    // the target, not the timed ones.  
	    if (((rolename == null) || (query == null)) && 
		!(layer.getLayerType() == NLayer.TIMED_LAYER) && 
		!(layer.inheritsTime())) {
		System.err.println("Warning:  -t tag should ordinarily be one that can have timings.");
	    }
	    // Get the attribute values, or names of elements at ends
	    // of role pointers.
	    List states = null;
	    states = new ArrayList();
	    if (attname != null) {
	    	if (attname.equalsIgnoreCase(controlData.getAgentAttributeName())) {
			    List agents = controlData.getAgents();
			    Iterator agent_it = agents.iterator();
			    while (agent_it.hasNext()) {
			    	NiteAgent agent = (NiteAgent) agent_it.next();
			    	String shortname = agent.getShortName();
			    	states.add((Object) shortname);
			    }
			} else {
				NAttribute att = tag.getAttributeByName(attname);
				if (att == null) {	
					System.err.println("Can't find attribute with name " + attname + " for tag with name " + tagname);
					System.exit(0);
				}
				if (att.getType() != NAttribute.ENUMERATED_ATTRIBUTE) {
					System.err.println("Attribute used must be of type enumerated.");
					System.exit(0);
				}	
				states = att.getEnumeratedValues();
			}
	    } else { //we're using the type of the thing at end of role
		List ptrlist = tag.getPointers();
		Iterator ptr_it = ptrlist.iterator();
		NPointer correct_ptr = null;
		while (ptr_it.hasNext()) {
		    NPointer ptr = (NPointer) ptr_it.next();
		    if (ptr.getRole().equals(rolename)) {
			correct_ptr = ptr;
		    }
		}
		if (correct_ptr == null) {
		    System.err.println("Named tag " + tagname + " doesn't have named role " + rolename + ". Exiting...");
		    System.exit(0);
		}
		String layername = correct_ptr.getTarget();
		NLayer targ_layer = controlData.getLayerByName(layername);
		if (targ_layer != null) { // found the target layer OK
		    List targ_els = targ_layer.getContentElements();
		    Iterator te_it = targ_els.iterator();
		    while (te_it.hasNext()) {
			NElement te = (NElement) te_it.next();
			String te_name = te.getName();
			states.add(te_name);
		    }
		} else {
		    // it must be that getTarget has returned an
		    // object set.
		    List objectsets = controlData.getObjectSets();
		    Iterator os_it = objectsets.iterator();
		    NObjectSet correct_os = null;
		    while (os_it.hasNext()) {
			NObjectSet os = (NObjectSet) os_it.next();
			if (os.getName().equals(layername)) {
			    correct_os = os;
			}
		    }
		    if (correct_os != null) {
			List targ_els = correct_os.getContentElements();
			Iterator te_it = targ_els.iterator();
			while (te_it.hasNext()) {
			    NElement te = (NElement) te_it.next();
			    String te_name = te.getName();
			    states.add(te_name);
			}
		    } else { // Barf; can't find tag values.
			// could be we're trying an ontology,
			// and they forgot the attribute name.
			System.err.println("Can't find the tag values by following the role to a set of elements as requested.  If the targets are in an ontology, name the significant attribute in -att and use the ontology element name for -tag, giving the timed elements in a query.  Exiting...");
			System.exit(0);
		    }
		}
	    }
	    // Call the n-gram size n, and the size of the enumerated list
	    // e.  We need an n-dimensional matrix of size e.  Then we
	    // keep a queue of integers 0 to e-1 of length n for the 
            // history of states in the ngram; for each query match, we
	    // update the queue and then write in the matrix using the
	    // values in the queue.
	    int e = states.size();
	    Integer nsize = Integer.valueOf(num);
	    int n = nsize.intValue();
	    // set up query if it doesn't exist
	    if (query == null) {
		query = "($t " + tagname + "):";
	    }
	    Hashtable counts = null;
	    // //Commented-out version separates log and error messages; putting the
	    // // log stuff to a file; it works.
	    //try {
	    //	FileOutputStream fileout = new FileOutputStream("logfile");
	    //	PrintStream msgs = new PrintStream(fileout);
	    try {
		// second arg sends log messages to System.err, not System.out.
		nom = new NOMWriteCorpus(controlData, System.err);
		nom.setLazyLoading(false);
		//nom = new NOMWriteCorpus(controlData, msgs);
		//nom.setErrorStream(msgs);
		counts = InitializeCounts(e, n);
		if (o != null) {
		    NObservation obs = controlData.findObservationWithName(o);
		    nom.loadData(obs);
		    counts = CalculateNGrams(nom, query, attname, rolename, n, states, counts, independent);
		} else { // do for each observation in turn
		    List obslist = controlData.getObservations();
		    for (int i = 0; i < obslist.size(); ++i) {
			NiteObservation nobs = (NiteObservation) obslist.get(i);
			nom.loadData(nobs);
		        counts = CalculateNGrams(nom, query, attname, rolename, n, states, counts, independent);
			// clearData reloads the corpus resources, which is a bit strange.
			nom.clearData();
		    };
		};
		PrintNGrams(states, counts, observationname, e, n);
	    } catch (NOMException nex) {
		nex.printStackTrace();
	    }
	} else {
	    System.err.println(
			       "This is a simple (one document) corpus: exiting...");
	}
	
    }


    /** Stupid, I know, but I can't figure out the syntax for
	power in Java and don't have a reference to hand.
    */
    private int power(int e, int n) {
	int e_to_the_n = 1;
	for (int i=0;i<n;++i) {
	    e_to_the_n = e_to_the_n * e;
	}
	return e_to_the_n;
    }

    // We're hashing on integers 0 to n*e, where n is the number
    // of the n-gram and e is the number of enumerated values.
    // Initialize hash table entries to 0.  

    private Hashtable InitializeCounts(int e, int n) {
	Hashtable counts = new Hashtable();
	int limit = power(e,n);
	for (int i=0;i<limit;++i) { 
	    Integer hash = new Integer(i);
	    Integer zero = new Integer(0);
	    counts.put((Object)hash,(Object)zero);
	}
	return counts;
    }


    /* given an array list with n entries of 0 to e-1, return the Integer
       used as the hash key for it.  It's the integer with 0 to e-1 for
       the first entry, e * (0 to e-1) for the second, e*e*(0 to e-1) for
       the third, etc. */
    private Integer hashOf (ArrayList history, int e, int n) {
	int value = 0;
	for (int i=0;i<n;++i) {
	    Integer hist_i = (Integer)history.get(i);
	    value = value + (power(e,i)*hist_i.intValue());
	}
	Integer newint = new Integer(value);
	return (newint);
    }

    /* given a Integer used as a hash key, construct the array list of
       n entries it was used for.  */
    private List unhash (Integer hash, int e, int n) {
	int remainder = hash.intValue();
	List ret_val = null;
	ret_val = new ArrayList();
	for (int i=n-1;i>-1; --i) {
	    int highest_fit = 0;
	    int limit = power(e,i);
	    for (int j=0;((j<e) && ((limit*j) <= remainder)); ++j) {
		highest_fit= j;
	    };
	    Integer newint = new Integer(highest_fit);
	    ret_val.add(newint);
	    remainder = remainder - (limit*highest_fit);
	}
	return (ret_val);
    }

    private void PrintNGrams(List states, Hashtable counts, String observationname, int e, int n) {
	int limit = power(e,n);
	for (int i=0;i<limit;++i) { 
	    Integer hash = new Integer(i);
	    Object val = counts.get((Object)hash);
	    Integer valnum = (Integer)val;
	    System.out.print(valnum.toString());
	    List history = unhash(hash, e, n);
	    for (int j=n-1;j>-1;--j) {
		System.out.print("\t" + reverseIndex((Integer)history.get(j),states));
	    }
	    System.out.print("\n");

	}
    }


    private Hashtable CalculateNGrams(NOMWriteCorpus nom, String q, String attname, String rolename, int n, List states, Hashtable counts, boolean independent) {
	Hashtable cts = counts;
	// evaluate the query;
	List outerelist = null;
	try {
	    outerelist = searchEngine.search(nom, q);
	} catch (Throwable e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	if (outerelist == null || outerelist.size() <= 1) {
	    System.err.println("NO QUERY MATCHES FOUND USING QUERY " + q + "!");
	    System.exit(0);
	}

	/* remove the first element of the result list that contains variable names */
	outerelist.remove(0);
        /* sort speaking turns in time order */
	Collections.sort(outerelist, mycomp);

	/* set up a queue of states for the ngrams */
	List history = null;
	history = new ArrayList();
	/* use this limited history to go through the query results in order, counting as we go */
	for (int i=1; i< outerelist.size(); i++) {
	    List outerreslist = (List) outerelist.get(i);
	    NOMElement el = (NOMElement) outerreslist.get(0);
	    String state = null;
	    if (rolename != null) {
		NOMPointer ptr = el.getPointerWithRole(rolename);
		NOMElement to_el = ptr.getToElement();
		// this is the correct value if using element names
		state = to_el.getName();
		// but change to attribute value if required
		if (attname != null) {
		    NOMAttribute att = to_el.getAttribute(attname);
		    if (att == null) {
			System.err.println("Element " + el.getID() + " points to element " + to_el.getID() + " using role " + rolename + " which doesn't have an attribute named " + attname + "; exiting");
			System.exit(0);
		    }
		    state = att.getStringValue();
		}
	    } else {
		if (attname == null) {
		    System.err.println("Got to impossible branch! Exiting...");
		    System.exit(0);
		}
		NOMAttribute att = el.getAttribute(attname);
		if (att == null) {
		    System.err.println("Found a match without a state value attribute! Exiting...");
		    System.exit(0);
		}
		state = att.getStringValue();
	    };
	    int index = IndexOf(state, states);
	    if (i < n) {
		/* We're still populating the history queue for the first n-gram - 
  		 add the item but don't count anything */
		Integer indint = new Integer(index);
		history.add((Object)indint);
	    } else {
		if (i > n) {
		    /* remove from the front of the queue before adding anything */
		    history.remove(0);
		}
		/* add the latest item to the queue */
		Integer indint = new Integer(index);
		history.add((Object)indint);
		/* if independent is false (so we're counting every n-gram in the normal
                   fashion) or if we're on a multiple of n, increment the count for this n-gram */
		int remainder = i;
		for (int j=i;j>0;j=j-n) {
		    remainder = j;	
		}
		if ((independent==false) || remainder == n) { 
		    Integer hash = hashOf((ArrayList)history, states.size(), n);
		    Object oldval = cts.get((Object) hash);
		    cts.remove((Object) hash);
		    Integer oldvalint = (Integer) oldval;
		    Integer newint = null;
		    if (oldvalint != null) {
			newint = new Integer(oldvalint.intValue() + 1);
		    } else {
			System.err.println("failed to create int");
		    }
		    cts.put((Object)hash, (Object)newint);
		}
	    }
	}
    return cts;
    }


    private int IndexOf(String state, List states) {
	int i = 0;
	while (i < states.size() && !(states.get(i).equals(state))) {
	    i = i + 1;
	}
	if (i == states.size()) {
	    System.err.println("Found a value not in the enumerated list!");
	    System.exit(0);
	} 
	return i;
    }

    /* given a number from 0 to e-1, find the corresponding string in 
       states */
    private String reverseIndex(Integer index, List states) {
	return ((String)states.get(index.intValue()));
    }


    /**
     * Called to start the  application. 
 * Call like this:
 *    java NGramCalc -corpus metadata_file_name -observation observation_name -tag tagname -query query -att attname -n positive_integer -role rolename -independent
 *
 * with corpus and tag required; observation optional; either an 
 * attribute name or a role or both; n defaults to 1; independent
 * defaults to false.
     */
    /** Start the application. */
    public static void main(String[] args) {
	String corpus=null;
	String observation=null;
	String tag = null;
	String query = null;
	String att = null;
	String role = null;
	boolean independent = false;
	String number = "1";
	if (args.length < 6 || args.length > 13) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-observation") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		observation=args[i];
	    } else if (flag.equals("-tag") || flag.equals("-t")) {
		i++; if (i>=args.length) { usage(); }
		tag=args[i];
	    } else if (flag.equals("-query") || flag.equals("-q")) {
		i++; if (i>=args.length) { usage(); }
		query=args[i];
	    } else if (flag.equals("-att") || flag.equals("-a")) {
		i++; if (i>=args.length) { usage(); }
		att=args[i];
	    } else if (flag.equals("-role") || flag.equals("-r")) {
		i++; if (i>=args.length) { usage(); }
		role=args[i];
	    } else if (flag.equals("-independent") || flag.equals("-i")) {
		independent=true;
	    } else if (flag.equals("-number") || flag.equals("-n")) {
		i++; if (i>=args.length) { usage(); }
		number=args[i];
	    } else {
		usage();
	    }
	}
	if ((corpus == null) || (tag == null)) { usage(); }
	if ((att == null) && (role == null)) { usage(); }
	if ((role != null) && (att != null) &&
	    (tag != null) && (query == null)) {
		System.err.println("Warning:  when role, att, and tag are used together, expect a query that matches elements of a type that points to the named tag using the role.");
	}
	NGramCalc m = new NGramCalc(corpus, observation, tag, query, att, role, number, independent);
    }

    private static void usage () {
	System.err.println("Usage: java NGramCalc -corpus <path-to-metadata> -observation <obsname> -tag <tagname> -att <attname> -query <query> -role <rolename> -number <number> -independent");
	System.exit(0);
    }
}

