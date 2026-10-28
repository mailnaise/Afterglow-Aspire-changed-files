#!/bin/bash

#declare -a JSS=("DG")
declare -a JSS=("GJ")

#### Adding seudo lightcurve for GW190425 ######
#declare -a INPUTDATAS=("afterglow_GW170817" "afterglow_rates_GW170817" "afterglow_rates_GW170817_GW190425" "seudo_afterglow_rates_GW170817_GW190425" "rates" "rates_GW170817" "rates_GW170817_GW190425" "GW170817" "GW190425" "GW170817_GW190425" )
#### Normal case ###
declare -a INPUTDATAS=( "GW170817" "afterglow_GW170817")
SAMPLER="${3:-aspire}" ## un-hardcoding the used sampler

for DATA in "${INPUTDATAS[@]}" ; do
for JS in "${JSS[@]}" ; do

bash submit_run.sh ${JS} ${1} ${DATA} ${2} ${SAMPLER}

done
done
