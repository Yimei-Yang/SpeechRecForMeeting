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
* FunctionQuery is a utility for outputting tab-delimited data.  It
* takes all elements resulting from the first matched variable of a
* query, as long as they are timed, and put them in order of start
* time.  Then it outputs one line per element the values of a the
* named attributes with a tab character between each attribute.
* 
* To call the utility, which is in builds after NXT 1.3.2 but otherwise
* available from CVS, set the classpath in the usual way, and then use 
* 
* java FunctionQuery -c metadata_file_name -ob observation_name -q query [-resources id+] -atts attname+
* 
* -corpus is required; it is the path to the metadata file.
* 
* -observation is optional: it's the name of the observation. If not
* present the program cycles through all observations.required;
* 
* -query is required; the first matched variable from every result
*   forms the basis of the output.
*
* -resources is optional; if it is supplied the list of resource ids
*   will be loaded using the resource information included in the
*   metadata file. If it is not included, the default resources are
*   loaded.
*
* -atts is required; input is expected as a space separated list of
    attributes. Note that if the attribute does now exist for some
    matched elements, a blank tab-stop will be the output.

    If the attribute starts with the '@' character, it will be
    interpreted as a function. Known functions are:
         @count(), @sum(), @extract(), @overlapduration()
    where: 
    @count(conquery) returns the number of results from evaluating
    "conquery" in the context of the current result of query,
    
    @sum(conquery, attr) returns the sum of the values of attr for all
    results of conquery evaluated in the context of query. Attr should be
    a numerical attribute.
    
    @extract(conquery, attr, n=0, last=n+1) returns the attr attribute of the nth
    result of conquery evaluated in the context of query. If n is less
    than 0, it returns the attr attribute of the nth last result. If
	last is provided, the attr value of all results whose index is at
	least n and less than last is returned. If last is less than 0, it
	will count back from the final result. If last equals zero, all items
	between n and the last result will be returned.
    
    @overlapduration(conquery) returns the length of time that the
    results of conquery overlap with the results of the main query.
    For some conquery results, this number may exceed the duration of
    the main query result. For example, the duration of speech for
    all participants over a period of time may exceed the duration of
    the time segment where there a multiple simultaneous speakers. This
    can be avoided, for example, by restricting the conquery to a
    specific agent.
* 
* ADDFUNCTION:
    To add a new function:
        Insert a new clause into EvaluateFunction() - see WhichFunction?
        Add a function to evaluate the new clause. Recommended signature:
            String MyFunction(List resultslist, String[] args)
            
        EvaluateFunction() contains two useful variables:
            innerreslist is the results of conquery for the current query result
            qf is the parsed form of the @function where:
                qf.name is the function name, stripped of the @ character
                qf.args is the list of arguments, where:
                    qf.args[0] is the string representing the conquery
                    qf.args[1] should be an attribute name if required
                    The rest of the arguments are free for any other use

* java FunctionQuery -corpus jcte1.metadata.xml -o jcte1_04 -q '($p phase)' -atts nite:id duration '@extract(($t trial):$p#$t, nite:id)' '@count(($t trial):$p#$t)' '@sum(($a action):$p#$a, duration)'
*
* will output a list of phases with their associated nite:id and durations
* as well as the id if the first trial that overlaps the current phase,
* a count of trials that overlap the current phase and the total duration
* of all actions that overlap the current phase.
*
* If you need to convert the tabbed output into csv format, consider a
* script like:
*
* sed 's/\t/\", \"/g' infile | sed 's/$/\"/' | sed 's/^/\"/' > out.csv 
*
* Based on SortedOutput.java
*
* @author Craig Nicol, Jean Carletta, Jonathan Kilgour May 2007
**/

public class FunctionQuery { 
    
    // Useful constants
    private String FN = "@";
    
    // the data 
    private NOMWriteCorpus nom;
    private NiteMetaData meta;

    private class query_function {
        String name;
        String[] args;
        
        public query_function(String n, String[] a) {
            name = n;
            args = a;
        }
        
        public query_function(String toparse) {
            // TODO: May need to sanitise input here...
            int bracketpos = toparse.indexOf("(");
            name = toparse.substring(FN.length(), bracketpos);
            String argstring = toparse.substring(bracketpos+1, toparse.length()-1);
            args = argstring.split(",");
            if (args[0].length() == 0) { // has to be valid query
                args=new String[0];
            }
            for (int i = 0; i < args.length; i++)
            {
                args[i] = args[i].trim();
            }
            //return "FN: " + fnname + ", ARG: " + args;
        }
        
        /*
        public String toString() {
            String out = FN + name = "(";
            for (String a: args) {
                out = out + a + ", ";
            }
            out = out.substring(0, out.length()-2) + ")";
            return out;
        } */
    }

    // the arguments passed in
    String corpusname;
    String observationname;
    String query;
    List atts;
    List resources;
    Boolean addtext;
    
    private Engine searchEngine = new Engine();
    private Comparator mycomp = new SearchResultTimeComparator();

    public FunctionQuery (String c, String o, String q, List a, List r, Boolean t) {

       corpusname = c;
       observationname = o;
       query = q;
       atts = a;
       addtext = t;
       resources = r;
       
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
        searchAndPrint(nom, observationname, query, atts, addtext);
        } else { // do for each observation in turn
        List obslist = meta.getObservations();
        for (int i = 0; i < obslist.size(); ++i) {
	    //System.err.println("REAPPLY RESOURCES");
	    // this is in the loop because clear data clears resource
	    // prefs - probably shouldn't!
	    for (Iterator rit = resources.iterator(); rit.hasNext(); ) {
		String res = (String) rit.next();
		System.err.println("Apply resource: " + res);
		nom.forceResourceLoad(res);
	    }
       
            NiteObservation nobs = (NiteObservation) obslist.get(i);
            nom.loadData(nobs);
            searchAndPrint(nom, nobs.getShortName(), query, atts, addtext);
            // clearData reloads the corpus resources, which is a bit strange.
            nom.clearData();
        };
        }
    } catch (NOMException nex) {
        nex.printStackTrace();
    }
    }

    /** Re-insert the empty lists filtered out of subresults **/
    /** Requires the lists to be sorted **/
    private List unfilterResults(List mainresults, List subresults) {
	//System.err.println("*** SUB ***" + subresults);
	//System.err.println("*** MAIN ***" + mainresults);
	if (subresults.size() >= mainresults.size()) {
	    return subresults;
	}
	for (int i = 0; i < mainresults.size(); ++i) {
	    List sub = java.util.Collections.EMPTY_LIST;
	    if (i < subresults.size()) {
		sub = new ArrayList((List)subresults.get(i));
		if (sub.size() > 0) {sub.remove(sub.size()-1);};
	    }
	    List main = new ArrayList((List)mainresults.get(i));
	    if (!sub.equals(main)) {
		//System.err.println("sub/main: " + sub + "/" + main);
		sub = main;
		sub.add(java.util.Collections.EMPTY_LIST);
		subresults.add(i, sub);
	    }
	}
	//System.err.println("### SUB ###" + subresults);
	//System.err.println("### MAIN ###" + mainresults);

	return subresults;
    }

    /** do the main work */
    private void searchAndPrint(NOMCorpus nom, String observationname, String q, List atts, Boolean addtext) {
        List[] sublists; // = new List[atts.size()];
	System.out.println("Found " + atts.size() + " args.");
        // This is for the $a variable lookup tables
        Hashtable[] varlists; // = new Hashtable[atts.size()];
        query_function qf;
        
        List vars=null;
        List reslist=null; 
        Hashtable myvars = new Hashtable();
       
        try {
            reslist = searchEngine.search((SearchableCorpus)nom, q);
            if (reslist.size()>0) {
                vars = (List)reslist.get(0);
                reslist.remove(0);
                Collections.sort(reslist, mycomp);

	       int i=0;
	       for (Iterator vit = vars.iterator(); vit.hasNext(); ) {
		  myvars.put(vit.next(), new Integer(i));
		  i++;
	       }
	    }
	   
        } catch (Throwable ex) {
            ex.printStackTrace();
            System.exit(0);
        }
        
        if (addtext.booleanValue()) {
	   System.err.println("WARNING: Use of -t / -text deprecated in FunctionQuery. Trying '" + (String)vars.get(0) + "'.");
	   atts.add((String)vars.get(0));
        }
        
       sublists = new List[atts.size()];
       varlists = new Hashtable[atts.size()];
       
        // VOID: If using named queries, check for this and copy them here
        for (int i = 0; i < atts.size(); i++) {
            String a = (String)atts.get(i);
            sublists[i] = java.util.Collections.EMPTY_LIST;
            varlists[i] = new Hashtable();
            
            if(a.startsWith(FN)) {
                try {
                    qf = new query_function(a);
                    if(qf.args.length > 0) {
                        System.err.println(q + "::" + qf.args[0]);
                        sublists[i] = searchEngine.search((SearchableCorpus)nom, q + "::" + qf.args[0]);
                                                
                        if (sublists[i].size()>0) {
                            List subreslist = (List)sublists[i].get(1);
                            List subquery = (List)subreslist.get(subreslist.size()-1);
                            vars = (List)subquery.get(0); 
                            sublists[i].remove(0); // sort on main query
                            Collections.sort(sublists[i], mycomp);
			    //System.err.println("pre Reslist/sublist: " + reslist.size() + "/" + sublists[i].size()); 
                        
			    sublists[i] = unfilterResults(reslist, sublists[i]);
			    //System.err.println("post Reslist/sublist: " + reslist.size() + "/" + sublists[i].size()); 
			   int h=0;
			   for (Iterator vit = vars.iterator(); vit.hasNext(); ) {
			      varlists[i].put(vit.next(), new Integer(h));
			      h++;
			   }
                        }
		       
                    }
                } catch (Throwable ex) {
                    ex.printStackTrace();
                    System.exit(0);
                }
            }
        }
       
        printResults(observationname, reslist, sublists, atts, myvars, varlists);
       
    }

   /* List PruneAndSort(List, Comparator)
    * Removes any leading variable names from the list
    * then sorts the remainder using the Comparator
    */
   private List PruneAndSort(List reslist, Comparator mycomp) 
     {
        //  Mostly the first item is ["$var1", "$var2", ...]
 	// but if the first result of the main query is repeated
	// we get a list that has already been stripped.
        //System.err.println(reslist.get(0));
	if (true) // reslist.size() > 0 && ((List)reslist.get(0)).get(0) instanceof String) 
	  {
	     reslist.remove(0);
	  }
	
        Collections.sort(reslist, mycomp);
	return reslist;
     }
   
   
    /* From CountQueryMulti.java
     * obsname is null when counting over whole corpus 
     */
    private void CountMatches(NOMCorpus nom, String q, String obsname) {
        List elist = null;
        try {
            elist = searchEngine.search((SearchableCorpus)nom, q);
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

    private Comparable getAttribute(NOMElement el, String attr)
    {
        // NOTE: We convert tabs and newlines to spaces to
        // preserved tab-delimited format
        if (attr == null) {
            String result = getTextRecursive(el);
            if (result != null) {
                result.replace('\t', ' ');
                result.replace('\n', ' ');
            }
            return (Comparable)result;
        }
        
        Comparable attrval = el.getAttributeComparableValue(attr);
        if (attrval==null) {
            if (attr.equalsIgnoreCase(meta.getIDAttributeName())) {
                return (Comparable)(el.getID());
            } else if  (attr.equalsIgnoreCase(meta.getStartTimeAttributeName())) {
                return (Comparable)(new Double(el.getStartTime()));
            } else if (attr.equalsIgnoreCase(meta.getEndTimeAttributeName())) {
                return (Comparable)(new Double(el.getEndTime()));
            } else if (attr.equalsIgnoreCase(meta.getAgentAttributeName())) {
                 return (Comparable)(el.getAgentName());
            } else {
                return (Comparable)("Not Found: " + el.getName() + "@" + attr);
            }
        } else {
            return attrval;
        }
    }
    
    /* split $w@attname into its constituents */
    private List splitClause(String clause) {
        String el=null;
        String att=null;
        List retlist = new ArrayList();
        int atsign = clause.indexOf("@");
        int strsign = clause.indexOf("$");
        if (atsign>0 && strsign>=0) {
            el = clause.substring(0,atsign);
            att = clause.substring(atsign+1, clause.length());
        } else if (atsign < 0 && strsign >= 0) {
            el = clause;
        } else { // atsign == 0 OR atsign == anything && strsign < 0
            att = clause;
        }
        retlist.add(el);
        retlist.add(att);
        return retlist;
    }

    /* return text of this element or any kids */
    private String getTextRecursive(NOMElement el)
    { 
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

    // GetQualifiedAttribute(reslist, attr, index, varnames)
    // Expects reslist in the form [($p, $q, ...), (El0p, El0q, ...), ...]
    // attr is of the form $a@att
    // Will return El[index][$a].att
    // i.e. the att attribute of the index'th element for variable $a
    // varnames is the lookup table for $a
    private Comparable getQualifiedAttribute(List reslist, String attr, int index, Hashtable varnames)
    {
        List clause = splitClause(attr);
        String elname = (String)clause.get(0);
        String attname = (String)clause.get(1);
        
        //index += 1; // As there is a header line
        
        Integer ind;
        if(elname==null) {
            ind = new Integer(0);
        } else {
            ind = (Integer)varnames.get(elname);
        }
        
        if (ind==null) {
            System.err.println("varnames size: " + varnames.size());
            System.err.println("Cannot find bound variable " + elname);
            System.exit(0);
        }
        
        List outerreslist = (List)reslist.get(index);
        if (outerreslist==null) { return (Comparable)"Index not found."; }
        NOMElement el = (NOMElement) outerreslist.get(ind.intValue());
        if (el==null) { return (Comparable)"No Element found."; }
        
        return getAttribute(el, attname);
    }
                
    // TODO: Would be great to abstract this function out
    private double CalculateSum(List reslist, String[] args, Hashtable varnames)
    {
        String attr = args[1];
        double sum = 0.0;
        if (reslist == null) { return 0.0; }
        
        for (int i=1; i< reslist.size(); i++) {
            Comparable attrval = getQualifiedAttribute(reslist, attr, i, varnames);
            //System.out.print(el.getName() + "@" + attr);
            //System.out.print(attrval.toString());
            try {
                sum += ((Double)attrval).doubleValue();
            } catch (ClassCastException e) {
                continue;
            }
        }
        return sum;
    }

    private double CalculateOverlapDuration(NOMElement el, List reslist, String[] args)
    {
        double els = el.getStartTime();
        double ele = el.getEndTime();
        
        double nexts = 0.0;
        double nexte = 0.0;
        
        double s = 0.0;
        double e = 0.0;
        double sum = 0.0;
        
        if (reslist == null) { return 0.0; }
        if (reslist.size() == 0) {return 0.0; }
        
        reslist = PruneAndSort(reslist, mycomp);
                
        for (int i=0; i< reslist.size(); i++) {
            List outerreslist = (List) reslist.get(i);
            if (outerreslist==null) { continue; }
            NOMElement next = (NOMElement) outerreslist.get(0);
            if (next==null) { continue; }
            
            nexts = next.getStartTime();
            nexte = next.getEndTime();
            
            if (nexts < els) { s = els; } else { s = nexts; }
            if (nexte > ele) { e = ele; } else { e = nexte; }
            
            sum += (e - s);
        }
        return sum;
        
    }

    // @extract(conquery, attr, [startindex, [endindex]])
    // Returns the value of attr (or the internal text if no attr)
    // for all conquery results where startindex <= index < endindex
    // If startindex is omitted, the first result is returned
    // If endindex is omitted, only the result at startindex is returned
    // If endindex is zero, all results from startindex to the end are returned
    // If startindex or endindex are negative, they count back from the end of the result list
    private String ExtractAttribute(List reslist, String[] args, Hashtable varnames)
    {
        if (reslist.size() == 0) { return "-"; }
    
        String attr = args[1];
        String value = "";
        Comparable attrval;
        int index = 0;
        int lastindex = 0;

       reslist = PruneAndSort(reslist, mycomp);
        
        if (args.length > 2) {
            index = new Integer(args[2]).intValue();
            if (index < 0) { index += reslist.size(); }
        }
        if (index < 0 || index >= reslist.size() ) { return "BadIndex: " + index; }
        lastindex = index + 1;
        
        if (args.length > 3) {
            lastindex = new Integer(args[3]).intValue();
            // If lastindex is provided and is zero
            // Return list to end
            if (lastindex <= 0) { lastindex += reslist.size(); }
        }
        if (lastindex < 0 || lastindex > reslist.size() ) { return "BadLastIndex: " + lastindex; }
        
        for(int i = index; i < lastindex; i++)
        {
            attrval = getQualifiedAttribute(reslist, attr, i, varnames);
            try {
                value += (String)attrval;
            } catch (ClassCastException e) {
                value += ((Double)(attrval)).toString();
            }
            value += " ";
        }
        return value;
    }

    // VOID: Add ability to use saved queries?
    //       This may do more harm than good
    //       e.g. if Query1 = ($w word)($a action):$w # $a
    //            allow sum(Query1, duration), count(Query1), ...
    // Given reslist over attr, return the result of evaluating
    // the function given by attr over the element in reslist
    // whose first element is el
    //
    // VOIDed since Unix allows:
    // export Q1 = "($w word)($a action):$w # $a
    //    then
    // java FunctionQuery ... -query "@sum($1, duration)"
    //    or
    // java FunctionQuery ... -query "@count($Q1"' && $a@agent="a")'
    private String EvaluateFunction(NOMElement el, List reslist, String attr, Hashtable varnames, int index) {
        query_function qf = new query_function(attr);
        
        if (qf.name.length() == 0 || qf.args.length == 0 || reslist == null || reslist.size() == 0) {
            return "-";
        }
        
        List innerreslist = java.util.Collections.EMPTY_LIST;
        
	// TODO: Need to index across all variables...
	// PruneAndSort is a temporary, buggy fix
	// Add NXT feature request to stop filtering Complex queries.

	/* This is for filtered results...
        for(int i = 0; i < reslist.size(); i++) {
            List thisresult = (List)reslist.get(i);
            if ((NOMElement)thisresult.get(0) == el) {
                innerreslist = (List)thisresult.get(thisresult.size()-1);
            }
        }
	//*/
        
	//* This is for unfiltered results.
	List thisresult = (List)reslist.get(index);
	innerreslist = (List)thisresult.get(thisresult.size()-1);
	//*/

        //return innerreslist.toString();
        //List innerreslist = (List)reslist.get(reslist.size()-1);
        
        // WhichFunction?
        if (qf.name.compareTo("sum") == 0) {
            return String.valueOf(CalculateSum(innerreslist, qf.args, varnames));
        } else if (qf.name.compareTo("extract") == 0) {
            return ExtractAttribute(innerreslist, qf.args, varnames);
        } else if (qf.name.compareTo("overlapduration") == 0) {
            return String.valueOf(CalculateOverlapDuration(el, innerreslist, qf.args));
        } else if (qf.name.compareTo("count") == 0) {
            // subquery includes header line, so remove one from
            // return size of subquery 
            return String.valueOf(java.lang.Math.max(innerreslist.size()-1, 0));
        }
        return "Unknown:"+qf.name+"?"+qf.args.length+"?"+qf.args[0]+"?";
    }

    private void printResults(String ob, List reslist, List[] sublists,  List atts, Hashtable myvars, Hashtable[] varlists) {
    System.out.println("Observation: " + ob + "; Result size: " + reslist.size() + "; atts: " + atts.size());
    for (int j=0; j<atts.size(); j++) {
        System.out.print((String)atts.get(j) + "\t");
    }
    System.out.print("\n");
    
    for (int i=0; i< reslist.size(); i++) {
        List outerreslist = (List) reslist.get(i);
        NOMElement el = (NOMElement) outerreslist.get(0);
        if (el==null) { continue; }
        for (int j=0; j<atts.size(); j++) {
            String attr = (String)atts.get(j);
            String val = "";
            if (attr.startsWith(FN)) {
                val = EvaluateFunction(el, (List) sublists[j], attr, varlists[j], i);
            } else {
                Comparable attrval = getQualifiedAttribute(reslist, attr, i, myvars);
                try {
                    val = (String)attrval;
                } catch (ClassCastException e) {
                    val = ((Double)(attrval)).toString();
                }
            }
            
            if (val!=null) {
                System.out.print(val);
            }
            System.out.print("\t");
        }
        
        System.out.println();
    }
    }

    /**
    * Called to start the  application. 
    * Call like this:
    *    java FunctionQuery -corpus metadata_file_name -observation observation_name -resources res1 res2 ... -query query -resources id1 id2 -atts attname1 attname2 .. 
    *
    * with corpus, query and atts required; observation and resources optional; 
    */
    public static void main(String[] args) {
    String corpus=null;
    String observation=null;
    String query = null;
    List atts = new ArrayList();
    List resources = new ArrayList();
    Boolean addtext = new Boolean(false);
    
    if (args.length < 6) { usage(); }
    for (int i=0; i<args.length; i++) {
        String flag=args[i];
        if (flag.equals("-corpus") || flag.equals("-c")) {
            i++; if (i>=args.length) { usage(); }
            corpus=args[i];
        } else if (flag.equals("-observation") || flag.equals("-o")) {
            i++; if (i>=args.length) { usage(); }
            observation=args[i];
        } else if (flag.equals("-query") || flag.equals("-q")) {
            i++; if (i>=args.length) { usage(); }
            query=args[i];
        } else if (flag.equals("-text") || flag.equals("-t")) {
            addtext = new Boolean(true);
	} else if (flag.equals("-resources") || flag.equals("-r")) {
            int j=i+1;
            while (j<args.length && !(args[j].startsWith("-"))) {
                resources.add(args[j++]);
            }        
	   if (j == args.length) {
		break;
	   } else {
	      i = j - 1;
	   }
        } else if (flag.equals("-atts") || flag.equals("-a")) {
            int j=i+1;
            while (j<args.length && !(args[j].startsWith("-"))) {
                atts.add(args[j++]);
            }
	   if (j == args.length) {
		break;
	   } else {
	      i = j - 1;
	   }
	   
        } else {
	   System.err.println("ERROR: Unknown option - " + flag);
            usage();
        }
    }
    if ((corpus == null) || (query == null)) { System.err.println("ERROR: -corpus and -query required."); usage(); }
    if (atts.size()==0) { System.err.println("ERROR: need at least one attribute to print."); usage(); }
        FunctionQuery m = new FunctionQuery(corpus, observation, query, atts, resources, addtext);
    }

    private static void usage () {
    System.err.println("Usage: java FunctionQuery -corpus <path-to-metadata> [-t] -observation <obsname> -query <query> [-resources <res1> [<res2> ...]] -atts <attname1> [<attname2> ...]  ");
    System.exit(0);
    }
}





