#!/usr/bin/env bash
set -e

############################
### BEGIN ARGUMENT SETUP ###
############################

old_dir=$(pwd)
cd /tmp


# Options
verbose=false
backend=''

# Colors
red='\e[0;31m'
nc='\e[0m' # Reset attributes
bold='\e[1m'
green='\e[32m'


# Module prefix
prefix=rnn-forecaster

# Package versions
weka_dl4j_version="1.4.1"
tsf_version="1.0.25"

ep="${bold}[${green}${prefix} build.sh${nc}${bold}]${nc}: "

### Check for color support ###
# check if stdout is a terminal...
if test -t 1; then

    # see if it supports colors...
    ncolors=$(tput colors)

    if test -n "$ncolors" && test $ncolors -ge 8; then #Enable colors
        ep="${bold}[${green}${prefix} build.sh${nc}${bold}]${nc}: "
    else #Disable colors
        ep="[${prefix} build.sh]: "
        bold=""
        nc=""
    fi
fi

function show_usage {
    echo -e "Usage: prepare.sh"
    echo -e ""
    echo -e "Optional arguments:"
    echo -e "   -v/--verbose            Enable verbose mode"
    echo -e "   -b/--backend            Select specific backend "
    echo -e "                           Available: ( CPU GPU )"
    echo -e "   -h/--help               Show this message"
    exit 0
}

### BEGIN parse arguments ###
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -v|--verbose)
    verbose=true
    shift # past argument
    ;;
    -b|--backend)
    backend="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    show_usage
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo -e "${ep}Parameters:"
echo -e "${ep}      verbose       = ${verbose}"
echo -e "${ep}      backend       = ${backend}"
echo -e "${ep}"
### END parse arguments ###

# Get platform
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     platform=linux;;
    Darwin*)    platform=macosx;;
    CYGWIN*)    platform=windows;;
    MINGW*)     platform=windows;;
    *)          platform="UNKNOWN:${unameOut}"
esac


if [[ ${backend} != 'CPU' && ${backend} != 'GPU' ]]; then
    echo -e "${ep}${red}Selected package must be either CPU or GPU!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
fi


# Package names
wdl4j_pn="wekaDeeplearning4j-${backend}-${weka_dl4j_version}-${platform}-x86_64"
tsf_pn="timeseriesForecasting${tsf_version}"

##########################
### END ARGUMENT SETUP ###
##########################


# Download and locally install the timeseries forecasting package
wget -O ${tsf_pn}.zip "http://prdownloads.sourceforge.net/weka/${tsf_pn}.zip?download"
unzip ${tsf_pn}.zip -d ${tsf_pn}
cd ${tsf_pn}
mvn -DskipTests -Dmaven.javadoc.skip=true install

cd ..

# Download and locally install the weka deeplearning4j package
wget "https://github.com/Waikato/wekaDeeplearning4j/releases/download/v${weka_dl4j_version}/${wdl4j_pn}.zip"
unzip ${wdl4j_pn}.zip -d ${wdl4j_pn}
cd ${wdl4j_pn}
mvn -DskipTests -P ${backend} install

cd ${old_dir}