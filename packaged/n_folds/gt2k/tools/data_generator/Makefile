all: gen_data 

gen_data: gen_data.c
	gcc gen_data.c -lgsl -lgslcblas -lm -o gen_data

clean: 
	rm gen_data 

