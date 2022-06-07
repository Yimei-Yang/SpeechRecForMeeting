import java.util.*;
import java.io.*;

// Import Serializer & Parser classes
import javax.xml.transform.*;
import javax.xml.transform.stream.*;
import javax.xml.transform.dom.*;
import javax.xml.parsers.*;
import org.apache.xml.serialize.*;
import org.jdom.JDOMException;
import org.jdom.input.*;
import org.jdom.output.DOMOutputter;
import org.jdom.transform.*;
import org.w3c.dom.Document;

// Import NITE stuff
import net.sourceforge.nite.meta.*;
import net.sourceforge.nite.meta.impl.*;
import net.sourceforge.nite.nom.nomwrite.impl.*;
import net.sourceforge.nite.nom.nomwrite.*;
import net.sourceforge.nite.nom.NOMException;

/**
 * Prepare a corpus for schema validation. This involves:
 * 1. Saving the corpus with no ranges using XLink / Xpointer syntax.
 * 2. Finding the generic schema files typelib.xsd and xlink.xsd
 * 3. Generating the data-specific schema file extension.xsd
 * @author Jonathan Kilgour, Feb 2003
 **/

public class PrepareSchemaValidation { 
    private static final String STYLESHEET_FILE = "generate-schema.xsl";
    private static final String KNIT_STYLESHEET_FILE = "knit.xsl";
    private static final String TYPE_SCHEMA_FILE = "typelib.xsd";
    private static final String XLINK_SCHEMA_FILE = "xlink.xsd";
    private static final String EXTENSION_SCHEMA_FILE = "extension.xsd";
    private static final String README_FILE = "README_SCHEMA_VALIDATION";
    

    public PrepareSchemaValidation(String c, String o) {
	NiteMetaData meta;
	NOMWriteCorpus nom;

	System.out.println("Preparing corpus " + c + " into directory " + o + ".");
	try {
	    File odir=new File(o);
	    if (!odir.exists()) {
		if (odir.mkdirs()) {
		    System.out.println("Made directory for output: " + o);
		} else {
		    System.err.println("ERROR: Failed to make directory for output: " + o);		
		    System.exit(0);
		}
	    }
	    meta = new NiteMetaData(c);
	    nom = new NOMWriteCorpus(meta);
	    nom.loadData();
	    nom.setSerializeMaximalRanges(false);
	    meta.setLinkType(NMetaData.XPOINTER_LINKS);
	    // Remember these paths are relative to metadata, and
	    // everything ends up in the same directory
	    meta.setCodingPath(".");
	    meta.setOntologyPath(".");
	    meta.setObjectSetPath(".");
	    meta.setCodingPath(".");
	    meta.setElementsAndAttributesToDefaults();
	    // the following because we can't have a namespace
	    // qualified attribute as an xsd:ID
	    meta.setIDAttributeName(NiteMetaConstants.noNamespaceID);
	    String mfn = o + File.separator + (new File(c)).getName();
	    meta.writeMetaData(mfn);
	    nom.setForceStreamElementNames(true);
	    nom.setSchemaLocation(EXTENSION_SCHEMA_FILE);
	    nom.serializeCorpus();
	    prepareStylesheets(o, mfn);
	    copyResources(o);
	} catch (NiteMetaException nme) {
	    System.err.println("Failed to load metadata file " + c);
	    System.exit(0);
	} catch (NOMException nex) {
	    System.err.println("NOM ERROR ");
	    nex.printStackTrace();
	    System.exit(0);
	} catch (Exception ex) {
	    ex.printStackTrace();
	}
    }

    private void prepareStylesheets(String o, String metadatapath) {

	// First create "extension.xsd" by passing the metadata file
	// through the "generate-schema.xsl" stylesheet
        InputStream in = ClassLoader.getSystemClassLoader().getResourceAsStream(STYLESHEET_FILE);	
        StreamSource xslstream1 = new StreamSource(in);
        try {
	    TransformerFactory tFactory = TransformerFactory.newInstance();
            Transformer trans1 = tFactory.newTransformer(xslstream1);

            // Get the XML input as DOM
            DocumentBuilderFactory dFactory = DocumentBuilderFactory.newInstance();
            dFactory.setNamespaceAware(true);
            DocumentBuilder dBuilder = dFactory.newDocumentBuilder();
            Document xmlInitDoc1 = dBuilder.parse(metadatapath);
            DOMSource xmlDomSource1 = new DOMSource(xmlInitDoc1);

            //Make a DOM result and do the transform
            DOMResult domResult1 = new DOMResult();
            trans1.transform(xmlDomSource1, domResult1);
            Document xmlDoc1 = (Document) domResult1.getNode();

	    String outfile=o + File.separator + EXTENSION_SCHEMA_FILE;
            SerializerFactory sf = SerializerFactory.getSerializerFactory("xml");
            Serializer s = sf.makeSerializer(new FileOutputStream(new File(outfile)), new OutputFormat());
            s.asDOMSerializer().serialize(xmlDoc1);
        } catch (TransformerConfigurationException tce) {
            //		tce.printStackTrace();
            System.out.println(
                "ERROR: transformer configuration error - failed to find XSL file "
                    + STYLESHEET_FILE);

        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
            System.out.println("A parser configuration error");
        } catch (org.xml.sax.SAXException se) {
            se.printStackTrace();
            System.out.println("A SAX error");
        } catch (TransformerException te) {
            System.out.println("A transformer error");
            te.printStackTrace();
        } catch (IOException ioe) {
            System.out.println("An io problem");
            ioe.printStackTrace();
        }
    }

    private void copyResources(String odir) {
	try {
	    InputStream instream = ClassLoader.getSystemClassLoader().getResourceAsStream(KNIT_STYLESHEET_FILE) ;	
	    File outfile = new File(odir + File.separator + KNIT_STYLESHEET_FILE);
	    copyFile(instream, outfile);
	    instream = ClassLoader.getSystemClassLoader().getResourceAsStream(TYPE_SCHEMA_FILE) ;	
	    outfile = new File(odir + File.separator + TYPE_SCHEMA_FILE);
	    copyFile(instream, outfile);
	    instream = ClassLoader.getSystemClassLoader().getResourceAsStream(XLINK_SCHEMA_FILE) ;	
	    outfile = new File(odir + File.separator + XLINK_SCHEMA_FILE);
	    copyFile(instream, outfile);
	    instream = ClassLoader.getSystemClassLoader().getResourceAsStream(README_FILE) ;	
	    outfile = new File(odir + File.separator + README_FILE);
	    copyFile(instream, outfile);
	} catch (IOException ioex) {

	}
    }

    // This method copies a given File into another one.
    // If the given File is a directory, it will recursively copy all subdirectories
    // and the files in them.
    private void copyFile(InputStream istream, File dest) throws IOException {
	OutputStream ostream = new FileOutputStream(dest);  
	dest.createNewFile();
	while(true){
	    int nextByte = istream.read();
	    if (nextByte==-1)
		break;
	    ostream.write(nextByte);
	}
	istream.close();
	ostream.close();
    }

    public static void main(String args[]){
	String corpus=null;
	String output_directory=".";

	if (args.length < 2 || args.length > 4) { usage(); }
	for (int i=0; i<args.length; i++) {
	    String flag=args[i];
	    if (flag.equals("-corpus") || flag.equals("-c")) {
		i++; if (i>=args.length) { usage(); }
		corpus=args[i];
	    } else if (flag.equals("-output") || flag.equals("-o")) {
		i++; if (i>=args.length) { usage(); }
		output_directory=args[i];
	    } else {
		usage();
	    }
	}
	if (corpus == null) { usage(); }
	
	PrepareSchemaValidation psv = new PrepareSchemaValidation (corpus, output_directory);
    }

    private static void usage () {
	System.err.println("Usage: java PrepareSchemaValidation -corpus <path-to-metadata> [ -o <output-directory> ]");
	System.exit(0);
    }

}
