#!/bin/bash

# Set this to the location you wish to save the data
OUT_PATH="$HOME/MSc/Afterglow_Aspire/Results/"

# Set this to the filepath to the activate file of your environment
ENVLOC="/cvmfs/software.igwn.org/conda/bin/activate"

# Set to the name of the environment if a conda environment, if not just set as ''
ENV='agve'


STRUCT=${1}
FILESTR=${2}

LOC="${OUT_PATH}AutoDoc/$(date +'%Y_%m_%d')/${FILESTR}/${3}_${STRUCT}/$(date +'%H_%M_%S')/"

IFS='_' read -r -a INDATAS <<< "${3}"

LFIXED="${4:-false}"
SAMPLER="${5:-aspire}"
NESSAI=true
INTERP=false
NDEFAULT=true
CORES=16

if ${LFIXED} ; then
LFIXED="--Lfixed"
else
LFIXED=""
fi

COMPILER="sample_model_nessai.py"

if ${INTERP} ; then
INTP="--interp"
else
INTP=""
fi

if ${NDEFAULT} ; then
DEF="--ndefault"
else
DEF=""
fi

if [ ! -r ${LOC} ]; then
    mkdir -p ${LOC}
fi

if [ "${LOC: -1}" != '/' ]; then
LOC="${LOC}/"
fi

BETA=1

EXT="temp/${STRUCT}/${BETA}/"
mkdir -p "${LOC}${EXT}"

for INDATA in "${INDATAS[@]}"
do
cp "./inputdata/${INDATA}"* "${LOC}."
cp "./inputdata/${INDATA}"* "${LOC}."
cp -r "./models/" "${COMPILER}" "${LOC}."
done

echo "universe = vanilla">"${LOC}${EXT}sample.sub"
echo "requirements = (Machine == \"wiay.astro.gla.ac.uk\")">>"${LOC}${EXT}sample.sub"
echo "log = ${LOC}${EXT}sample.log">>"${LOC}${EXT}sample.sub"
echo "error = ${LOC}${EXT}sample.err">>"${LOC}${EXT}sample.sub"
echo "output = ${LOC}${EXT}sample.out">>"${LOC}${EXT}sample.sub"
echo "executable = ${PWD}/wrapper.sh">>"${LOC}${EXT}sample.sub"
echo "arguments = ${ENVLOC} ${ENV} ${LOC}${COMPILER} --path ${LOC} --beta ${BETA} --struct ${STRUCT} --draws 1000 ${DEF} --cores ${CORES} --burn 2000 ${INTP} ${LFIXED} --sampler ${SAMPLER}">>"${LOC}${EXT}sample.sub"
echo "accounting_group = ligo.sim.o3.cbc.bayesianpopulations.parametric">>"${LOC}${EXT}sample.sub"
echo "transfer_input_files = ">>"${LOC}${EXT}sample.sub"
echo "request_cpus = ${CORES}">>"${LOC}${EXT}sample.sub"
echo "request_memory = 8GB">>"${LOC}${EXT}sample.sub"
echo "request_disk = 3GB">>"${LOC}${EXT}sample.sub"
echo "queue 1">>"${LOC}${EXT}sample.sub"
condor_submit "${LOC}${EXT}sample.sub"

#${PWD}/wrapper.sh ${ENVLOC} ${ENV} ${LOC}${COMPILER} --path ${LOC} --beta ${BETA} --struct ${STRUCT} --draws 1000 ${DEF} --cores ${CORES} --burn 2000 ${INTP}

exit 0
