EXECUTABLES=prepare create_grammar.pl random check_opts.sh htk-prep recognize.sh train.sh new_project.sh
INST=$(addsuffix .inst,$(EXECUTABLES))

CC=g++
GLOBALOPT=-O3
all:random htk-prep htkhelper.o prepare

random:random.cc
	$(CC) $(GLOBALOPT) random.cc -o random -lm

htk-prep:htk-prep-amin.c
	$(CC) $(GLOBALOPT) htk-prep-amin.c -o htk-prep -lm

htkhelper.o: htkhelper.cc htkhelper.h
	$(CC) htkhelper.cc -c -I.

prepare: standalone_prepare.c htkhelper.o 
	$(CC) standalone_prepare.c htkhelper.o -o prepare

install: $(INST)

%.inst:
	install -D -b $* $(INST_DIR)/$*

clean:
	rm -f htk-prep-amin random *.o prepare htk-prep

