CC 	  = g++
INC 	  = -I/usr/local/include -I/usr/include 

CFLAGS 	  = -Wall -O2 -g -pedantic -D_REENTRANT $(INC)
LIBS	  = -L/usr/lib -lm -lpthread -lfl

EXE 	  = htk2dot
SRC       = htk2dot.c
HFILES    = $(SRC:.c=.h)
OBJ	  = $(SRC:.c=.o)
DEPS	  = Makefile
STRIP 	  = /usr/bin/strip
RM 	  = /bin/rm



all: $(EXE)

debug:
	$(CC) -DDEBUG $(CFLAGS) $(LIBS) -o $(EXE) $(SRC)


debug-func:
	$(CC) -DFUNC_DEBUG $(CFLAGS) $(LIBS) -o $(EXE) $(SRC)


$(EXE): $(HFILES) $(OBJ) $(DEPS)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE) $(LIBS)

.c.o:	$(HFILES)
	$(CC) -c $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) -f *.o $(EXE) core *~

install:
	$(STRIP) $(EXE)
#======================================
