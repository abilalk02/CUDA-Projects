
all: AerosolModel


AerosolModel: Aerosol.cu
	nvcc -o Aerosol Aerosol.cu -lcudart -lcuda -lm

clean:
	$(RM) Aerosol


run: run_AerosolModel

run_AerosolModel:
	./Aerosol
	

