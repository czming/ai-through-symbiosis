/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "htk2dot.h"

#define TRANSITION_MATRIX_TAG "<TransP>"

const int MAX_LABEL_LEN	= 128;

// *********************************************************************
// given an HMM definition file in HTK format, generate graphviz-DOT
// compliant specification file
// *********************************************************************
int main(int argc, char *argv[])
{
    trans_t matrix;
	
    matrix.num_states   = 0;
    matrix.num_entries  = 0;
    matrix.entries      = NULL;
    
    load_transition_def( &matrix, stdin );
    generate_dot_file( &matrix, stdout );
}


// *********************************************************************
// given data about a transision matrix, generate a text file that can
// be interpreted by DOT
// *********************************************************************
void generate_dot_file( trans_t* matrix, FILE* stream )
{
    
    int row 		= 0;
    int col 		= 0;
    int i   		= 0;

    // the entries in the matrix indicate transition probabilities
    // the rows represent the current state and the columns represent
    // transitions states.  so given a coordiante (R,C) of ( 2, 3 ) this
    // would provide the probability of transitioning from state 2 into
    // state 3
    
    if( stream != NULL)
    {
	// print the DOT syntax for the name of the graph and orient
	// the graph to draw from left to right
	fprintf( stream, " digraph HMM_visualization {\n " );
	fprintf( stream, "  graph [rankdir=\"LR\"] \n " );

	// write all arrows going to and from state X
	for( row = 0; row < matrix->num_states; row++ )
	{
	    for( col = 0; col < matrix->num_states; col++ )
	    {
//		fprintf( stderr,
//			 "%d %d %f %d \n",
//			 row, col, matrix->entries[i], matrix->num_entries );
		    

		// if we have a nonzero transition probability then
		// draw the node and the arc
		if( matrix->entries[i] > 0.0f )
		{
		    char label[MAX_LABEL_LEN];

		    // convert the entry value into an edge label
		    pretty_print_sprintf(label, matrix->entries[i], 5);

		    // output the DOT syntax for the transition probabilities
		    fprintf( stream,
			     "%d -> %d [label=\"%s\", fontsize=10]; \n",
			     row, col, label );

		}

		// increment the entries index
		i++;
	    }
	    
	    // output the DOT syntax for characteristics of the state
	    fprintf( stream,
		     "%d [color=black, fillcolor=grey, style=filled]; \n",
		     row );
	}

	// print out the DOT syntax for the end of the DOT file
	fprintf( stream, "} \n" );

	
	// make sure we didn't have an error
	CHK_FERROR( stderr, "Errors occured while writing to the stream" );

    }	// END --> if NULL stream
    
}


// *********************************************************************
// Read in an HMM definition file and store the information about the
// transition matrix.  All other information is ignored
// *********************************************************************
void load_transition_def( trans_t* matrix, FILE* stream )
{
    if( stream != NULL)
    {
	const int MAX_DATA_STR 		= 1000;
	
	int	  elements_read		= 0;
	char	  data_str[MAX_DATA_STR];
	int	  str_len 		= 0;

	// specific varaibles for this parser
	flag_t    transition_tag_found  = UNSET;
	flag_t    num_states_found      = UNSET;
	flag_t	  definition_complete	= UNSET;

	int matrix_elements_read	= 0;
	
	
	// data values are stored in the infile stream as strings.  read in each
	// string representing the data value and convert it to the appropriate
	// data value.  Do this until either the end of the stream is reached,
	// the storage buffer is exceeded, or the end of the line is reached.
	while( (!feof( stream ) ) && (str_len < MAX_DATA_STR) &&
	       (definition_complete == UNSET) )
	{
	    int char_read;
	    
	    // read in a single character from the input stream
	    char_read = fgetc( stream );

	    // if that character is not whitespace add it to our current string
	    if(  !isspace( char_read ) )
	    {
		data_str[str_len] = char_read;
		str_len++;		
	    }
	    else
	    {
		// we have just read a character of whitesace which means we are
		// we have either (a) just completed reading a string (b) we are
		// eating whitespace inbetween data values.

		// if our string has length, then this whitespace signifies the
		// end of a token, otherwise it is just white space between
		// tokens that should be ignored
		if( str_len > 0 )
		{
		    // make sure we have not exceeded the size of our buffer
		    str_len = 	( str_len > MAX_DATA_STR ) ?
				MAX_DATA_STR : str_len;
		    
		    // insert the NULL terminator into the string
		    // just to be on the safe side since we are not
		    // clearing the buffer after each token.
		    data_str[str_len] = '\0';

		    
//		    fprintf(stderr," Read Token: %s \n", data_str );
		    
		    // now that we have a token figure out what to do with it
		    if( strcasecmp( data_str, TRANSITION_MATRIX_TAG ) == 0 )
		    {
			transition_tag_found  = SET;
		    }
		    // see if we are reading in the definition for the matrix
		    else if( transition_tag_found == SET )
		    {
			// if we have read in the tag, then we are reading in
			// data that we care about

			// first we need to read in the number of states
			if( num_states_found == UNSET )
			{
			    matrix->num_states = atoi( data_str );
			    num_states_found = SET;
			    matrix->num_entries =
				matrix->num_states * matrix->num_states;
			}
			// next we need to read in the transition matrix
			// one row at a time
			else
			{
			    // if we have not yet created the matrix create it
			    if( matrix->entries == NULL )
			    {
				MALLOC_WEC( matrix->entries, *(matrix->entries),
					    matrix->num_entries );
			    }

			    // if we have not read in the entire matrix, then
			    // store the current token in the matrix and
			    // update the element count
			    if( matrix_elements_read < matrix->num_entries )
			    {
				matrix->entries[ matrix_elements_read ] =
				    atof( data_str );

				matrix_elements_read++;
				
			    }
			    // if all of the elements of the matrix are read
			    // in then we have completed the definition
			    else
			    {
				definition_complete = SET;
			    }
			    
			}			
			
		    }
		    
		    
		    // indicate that we have successfully converted and stored
		    // a data token and reset the string length to zero
		    elements_read++;
		    str_len = 0;
		}		
		
	    }	// END --> else: if we have a char of whitespace
	}	// END --> while: reading file
	

	
	// make sure we didn't have an error
	CHK_FERROR( stream, "Errors occured while reading in the stream" );

    }	// END --> if NULL stream
    
}


// *********************************************************************
// 
// *********************************************************************
int pretty_print_sprintf(char* fp, float v, int width)
{
   char s[MAX_LABEL_LEN];
   int i, len;

   // print the number with the max number of decimals
   if (sizeof(float) == sizeof(double)) sprintf(s, "%0.11f", v);
   else sprintf(s, "%0.6f", v);

   // find the end and count back over 0s and .
   for(i=0; s[i]; i++);
   for(i--; s[i]=='0' || s[i]=='.'; i--)
   {
      if (s[i]=='.')
      {
         i++;
         if (!s[i]) s[i] = '0';
         break;
      }
   }
   s[i+1] = '\0';
   if (strcasecmp(s, "-0")==0) strcpy(s, "0");
   len = strlen(s);
   if (fp)
   {
      i = len;
      while(i<width)
      {
         //memmove(&s[1], &s[0], sizeof(char) * (i+1));
         //s[0] = ' ';
         s[i] = '0';
         i++;
         s[i] = '\0';
      }
      sprintf(fp, "%s", s);
   }
   return len;
}
