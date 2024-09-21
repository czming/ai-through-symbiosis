# Main Makefile for GT2k.  Compiles and installs our utilities and tools,
# as well as HTK, if so desired.  Edit these things appropriately, and 
# be sure to have permissions for writing in all of the install directories!!

# install HTK (1) or don't (0)
INSTALL_HTK=0

# where the HTK source is
HTK_LOC=./HTK-3.2.tar.gz

# where the HTK binaries will go
HTK_DEST=/usr/local/bin	

# where the gt2k scripts and examples will go
GT2K_DEST=/usr/local/gt2k

###################################################################
SUBDIRS=utils

MAKEDIRS=$(addsuffix .all,$(SUBDIRS))

COPYDIRS=tools template tutorials examples

INSTDIRS=$(addsuffix .inst,$(COPYDIRS))

COPYFILES=LICENSE README PublicUse.doc PublicUse.txt 

INSTFILES=$(addsuffix .inst,$(COPYFILES))

all:	htk $(MAKEDIRS)

%.all:	
	make -C $*

htk:
	if [ $(INSTALL_HTK) == 1 ]; then ./install-htk.sh $(HTK_LOC) $(HTK_DEST); fi

install: utils/random utils/prepare $(INSTDIRS) $(INSTFILES)
	cd utils/; make "INST_DIR=$(GT2K_DEST)/utils" install

%.inst:
	cp -r $* $(GT2K_DEST)
