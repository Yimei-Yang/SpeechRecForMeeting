

import java.io.PrintWriter;

import java.util.HashMap;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;


import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.NamedNodeMap;

import org.xml.sax.SAXException;


import org.apache.xml.serialize.OutputFormat;
import org.apache.xml.serialize.XMLSerializer;
import org.apache.xml.serialize.LineSeparator;

import java.io.IOException;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.FactoryConfigurationError;
import javax.xml.parsers.ParserConfigurationException;

/**
 * Translator to take EventEditor output and make NXT input. 
 * 
 * At 01 June 05, the translator assumes that there is only
 * one group of tags in the EventEditor output, and so ignores
 * anything about the group structure.  This may suffice for
 * our purposes, but it means the translator isn't generic.
 * To make it generic, we would probably want to get rid of the
 * command line specification for tagname (since there could be
 * several) and construct that somehow from the group name.
 * 
 * For NXT it's useful to have an end time for the last tag,
 * but EventEditor gives no way of retrieving this.  We either
 * need to leave it off the data, get it from the video signal
 * used to code the data in the first place, or ask for changes
 * to EventEditor, either to put the end time for the video in the
 * output header, or to put explicit end times on each event.
 * 
 * We also should write ids on the elements, and the namespace
 * declaration for nite:, or change root to something besides
 * nite:root.
 * 
 * Jean Carletta, 01 June 05
 */

public class EventEditorToNXT {

    // special case Strings for nods and shakes
    private static final String FORM="form";

    /**
     * new document for output with a root tag nite:root.
     * 
     * @return
     */

    private static Document createOutputDocument() {
	Document document = null;
	try {
	    DocumentBuilderFactory factory = 
		DocumentBuilderFactory.newInstance();
	    DocumentBuilder builder = factory.newDocumentBuilder();
	    document = builder.newDocument();
	} catch (FactoryConfigurationError e) {
	    // unable to get a document builder factory
	} catch (ParserConfigurationException e) {
	    // parser was unable to be configured
	}
	Node root = (Node) document.createElement("nite:root");
	((Element)root).setAttribute("xmlns:nite", "http://nite.sourceforge.net/");
	document.appendChild(root);
	return document;
    }

    private static Document parseFile(String infile) {

	Document document = null;
	try {
	    DocumentBuilderFactory factory = DocumentBuilderFactory
		.newInstance();
	    DocumentBuilder builder = factory.newDocumentBuilder();
	    document = builder.parse(infile);
	} catch (FactoryConfigurationError e) {
	    // unable to get a document builder factory
	} catch (ParserConfigurationException e) {
	    // parser was unable to be configured
	} catch (SAXException e) {
	    // parsing error
	} catch (IOException e) {
	    // i/o error
	}
	if (document == null) {
	    System.err.println("No document for file " + infile + "!");
	    System.exit(0);
	}
	return document;
    }

    /**
     * Returns the CData contained within the only child of the passed element
     * with the given tag name, or null (without complaint) if there is no such
     * only child or there is an only child but no cdata.
     * 
     * @param el
     * @param tagname
     * @return
     */
    private static String getCDataFromOnlyChildWithName(Element el,
							String tagname) {
	String retval = null;
	NodeList childrenwithname = el.getElementsByTagName(tagname);
	if (childrenwithname.getLength() == 1) {
	    Node firstchild = (Node) childrenwithname.item(0);
	    NodeList cdata_nodes = firstchild.getChildNodes();
	    if (cdata_nodes.getLength() == 1) {
		Node cdata = (Node) cdata_nodes.item(0);
		try {
		    retval = cdata.getNodeValue();
		} catch (Exception e) {
		    System.err
			.println("Problem getting textual content of cdata for child with name"
				 + tagname + "!");
		    System.exit(0);
		}

	    }
	}
	return retval;
    }

    /**
     * In EventEditor output, Type elements contain ID elements, a number
     * starting counting from 0, and Name elements, a string that is the type
     * indexed by that number. Set up a table lookup from the number to the
     * string.
     * 
     * @param document
     * @return
     */
    private static HashMap setUpTypeMap(Document document) {
	HashMap typemap = new HashMap();
	NodeList elements = document.getElementsByTagName("Type");
	if (elements == null) {
	    System.err.println("No types defined!");
	    System.exit(0);
	}
	int elementCount = elements.getLength();
	for (int i = 0; i < elementCount; i++) {
	    Element typeelement = (Element) elements.item(i);
	    String idstring = getCDataFromOnlyChildWithName(typeelement, "ID");
	    String namestring = getCDataFromOnlyChildWithName(typeelement,
							      "Name");
	    typemap.put(idstring, namestring);
	}
	return typemap;
    }

    /** Create a node in the output NXT document corresponding to one event
     * from EventEditor's representation.
     * 
     * @param outdoc
     * @param typestring
     * @param starttime
     * @param textstring
     * @param tag
     * @param att
     * @param start
     * @param comment
     * @return

     */
    private static Element processEvent(Document outdoc, String typestring,
			String starttime, String textstring, List parameters, 
			String tag, String att,	String start, String comment) {
	//System.out.println(typestring + " \t" + starttime + "\t " + textstring);
	Node root = outdoc.getDocumentElement();
	Element newnode = outdoc.createElement(tag);
	newnode.setAttribute(att, typestring);
	newnode.setAttribute(start, starttime);
	if (textstring != null) {
	    newnode.setAttribute(comment, textstring);
	}
	if (parameters!=null) {
	    for (Iterator pit=parameters.iterator(); pit.hasNext(); ) {
		Node attr = (Node)pit.next();
		newnode.setAttribute(attr.getNodeName(), attr.getNodeValue());
	    }
	}
	    
	root.appendChild(newnode);
	return newnode;
    }
    
    /** 
     * @param el
     * @param endtime
     * @param end
     */
    private static void addEndToPreviousEvent(Element el, String endtime,
					      String end) {
	el.setAttribute(end, endtime);
    }

    /**
     * go through the event tags one by one, printing output. Be careful to only
     * do Event tags under the File tag, since there are other ones that mean
     * something else.
     * 
     * @param document
     * @param typemap
     * @param tag
     * @param att
     * @param start
     * @param end
     */
    private static void processEventTags(Document document, Document outdoc,
		 HashMap typemap, String tag, String att, String start, String end,
		 String comment, Double endtime) {
	NodeList fileelements = document.getElementsByTagName("File");
	if (fileelements.getLength() != 1) {
	    System.err.println("More or less than one file element in input!");
	    System.exit(0);
	}
	Element firstfile = (Element) fileelements.item(0);
	NodeList elements = firstfile.getElementsByTagName("Event");
	if (elements == null) {
	    System.err.println("No coding present!");
	    System.exit(0);
	}
	int elementCount = elements.getLength();
	Element new_el = null;
	for (int i = 0; i < elementCount; i++) {
	    Element eventelement = (Element) elements.item(i);
	    String idstring = getCDataFromOnlyChildWithName(eventelement, "ID");
	    String timestring = getCDataFromOnlyChildWithName(eventelement,
							      "Time");
	    String textstring = null;
	    if (eventelement.getElementsByTagName("Text").getLength() > 0) {
		textstring = getCDataFromOnlyChildWithName(eventelement, "Text");
	    }
	    
	    List paramlist = null;
	    if (eventelement.getElementsByTagName("Parameters").getLength() > 0) {
		Element paramelement = (Element) eventelement.getElementsByTagName("Parameters").item(0);
		//System.err.println("Get parameters!");
		paramlist = new ArrayList();
		NamedNodeMap nnm = paramelement.getAttributes();
		if (nnm!=null && nnm.getLength()>0) {
		    for (int ii=0; ii<nnm.getLength(); ii++) {
			Node node = nnm.item(ii);
			if (node.getNodeName().equalsIgnoreCase(FORM)) {
			    textstring=node.getNodeValue();
			} else {
			    paramlist.add(node);
			}
		    }
		}
	    }
	    // EventEditor gives times in ms; convert to seconds.
	    String starttimestring = null;
	    try {
		double starttime = Double.valueOf(timestring.trim()).doubleValue();
		starttime = starttime / 1000;
		starttimestring = Double.toString(starttime);
	    } catch (NumberFormatException nfe) {
		System.out.println("NumberFormatException: " + nfe.getMessage());
	    }
	    // from EventEditor format, we can only know what the end time for
	    // an element is once we have processed the next one. Keep a
	    // pointer to the previous element, and once we're past the first
	    // one, add the end time after the fact.
	    // This won't add an end time for the last element.
	    if (i > 0) {
		addEndToPreviousEvent(new_el, starttimestring, end);
	    }
	    new_el = processEvent(outdoc, (String) typemap.get(idstring),
				  starttimestring, textstring, paramlist, 
				  tag, att, start, comment);
	    
	}
	
	// fill in the end time of the last element if we have one.
	if (elementCount > 0 && endtime!=null) {
	    addEndToPreviousEvent(new_el, endtime.toString(), end);
	}
	
    }

    private static void serializeOutput(PrintWriter out, Document document) {
	//OutputFormat format = new OutputFormat((Document) core);
	OutputFormat format = new OutputFormat();
	format.setLineSeparator(LineSeparator.Windows);
	format.setIndenting(true);
	format.setLineWidth(0);
	format.setPreserveSpace(true);
	//XMLSerializer serializer = new XMLSerializer(new FileWriter(
	//		"output.xml"), format);
	XMLSerializer serializer = new XMLSerializer(out, format);
	OutputFormat outputformat = new OutputFormat();
	outputformat.setIndenting(true);
	serializer.setOutputFormat(outputformat);
	try {
	    serializer.asDOMSerializer();
	    serializer.serialize(document);
	} catch (Exception e) {
	    System.out.println("Failure to serialize!");
	}
	
    }

    private static void convert(String infile, String tag, String att,
				String start, String end, String comment, Double endtime) {
	//ParserWrapper parser = setUpParser();
	//Document document = parseFile(parser, infile);
	Document document = parseFile(infile);
	Document outdoc = createOutputDocument();
	HashMap typemap = setUpTypeMap(document);
	processEventTags(document, outdoc, typemap, tag, att, start, end,
			 comment, endtime);
	PrintWriter out = new PrintWriter(System.out);
	serializeOutput(out, outdoc);
    }

    /**
     * Called to start the application. Legal command line arguments are:
     * <ul>
     * <li>-infile input_filename</li>
     * <li>-tag tag_name_to_use_for_output_per_code</li>
     * <li>-att att_name_on_code_for_the_type</li>
     * <li>-start start_time_attribute</li>
     * <li>-end end_time_attribute</li>
     * <li>-last final_time_for_end_of_last_element (optional)</li>
     * </ul>
     *  
     */
    public static void main(String[] args) {
	String infile = null;
	String tag = null;
	String att = null;
	String start = null;
	String end = null;
	String comment = null;
	Double finaltime = null;

	if (args.length < 12 || args.length > 14) {
	    usage();
	}
	for (int i = 0; i < args.length; i++) {
	    String flag = args[i];
	    if (flag.equals("-infile") || flag.equals("-i")) {
		i++;
		if (i >= args.length) { usage(); }
		infile = args[i];
	    } else if (flag.equals("-tag") || flag.equals("-t")) {
		i++;
		if (i >= args.length) { usage(); }
		tag = args[i];
	    } else if (flag.equals("-att") || flag.equals("-a")) {
		i++;
		if (i >= args.length) { usage(); }
		att = args[i];
	    } else if (flag.equals("-start") || flag.equals("-s")) {
		i++;
		if (i >= args.length) { usage(); }
		start = args[i];
	    } else if (flag.equals("-end") || flag.equals("-e")) {
		i++;
		if (i >= args.length) { usage(); }
		end = args[i];
	    } else if (flag.equals("-comment") || flag.equals("-c")) {
		i++;
		if (i >= args.length) { usage(); }
		comment = args[i];
	    } else if (flag.equals("-last") || flag.equals("-l")) {
		i++;
		if (i >= args.length) { usage(); }
		try {
		    finaltime = new Double(args[i]);
		} catch (NumberFormatException nex) {
		    System.err.println("Final time is not a valid number: " + args[i]);
		    usage();
		}		
	    } else {
		usage();
	    }
	}
	if (infile == null) {
	    usage();
	}
	
	EventEditorToNXT.convert(infile, tag, att, start, end, comment, finaltime);
    }

    private static void usage() {
	System.err
	    .println("Usage: java EventEditorToNXT -i input-filename -t tag -a attribute -s start_attribute -e end_attribute -c comment_attribute [-l endtime]");
	System.exit(0);
    }
}
