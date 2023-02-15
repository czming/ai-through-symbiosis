/*
* All code in the project is provided under the terms specified in
* the file "Public Use.doc" (plaintext version in "Public Use.txt").
*
* If a copy of this license was not provided, please send email to
* haileris@cc.gatech.edu
*/
#ifndef __tlw_htk2dot_h__
#define __tlw_htk2dot_h__
/************************************************************************/
/*									*/
/************************************************************************/

// check to see if an error occured when manipulating a FILE* stream
#define CHK_FERROR( stream, msg )			\
	if( ferror( (stream) ) ) printf("%s \n", msg)


// hack so that C++ will be happy with MALLOC macros
#ifdef __cplusplus
#define CAST(ptr)		( typeof((ptr)) )
#else
#define CAST(ptr)
#endif

#define MALLOC_WEC( ptr, ptr_type, amount ) 				\
       if( (ptr = CAST(ptr)malloc( ( sizeof(ptr_type) * amount ) ) ) == NULL)\
	{			       					\
	    perror( "Malloc Failure \n" );				\
	    exit(EXIT_FAILURE);						\
	}			  


typedef enum{ UNSET, SET }flag_t;


typedef struct _trans_t
{
    int 	num_states;
    int		num_entries;
    float* 	entries;    

}trans_t;


void load_transition_def( trans_t* matrix, FILE* stream );
void generate_dot_file( trans_t* matrix, FILE* stream );
int pretty_print_sprintf(char* fp, float v, int width);

#endif
