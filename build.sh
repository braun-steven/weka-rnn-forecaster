#!/usr/bin/env bash

# Options
install_pack=false
verbose=false
clean=false
out=/dev/null
backend=''

# Colors
red='\e[0;31m'
nc='\e[0m' # Reset attributes
bold='\e[1m'
green='\e[32m'


# Module prefix
prefix=rnn-forecaster


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

# Project version (TODO: Fix for non GNU grep versions)
version=`grep -Po 'name="version" value="\K([0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta)\.[0-9]+)?)(?=")' build_package.xml`
if echo ${version} | grep -Eq "^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta)\.[0-9+])?$"; then
    echo -e "${ep}Building version: ${version}"
else
    echo -e "${ep}Error finding version. Unknown version: ${version}"
    echo -e "${ep}Exiting now."
    exit 1
fi

function show_usage {
    echo -e "Usage: build.sh"
    echo -e ""
    echo -e "Optional arguments:"
    echo -e "   -v/--verbose            Enable verbose mode"
    echo -e "   -i/--install-package    Install selected package"
    echo -e "   -c/--clean              Clean up build-environment"
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
    -i|--install-package)
    install_pack=true
    shift # past argument
    ;;
    -c|--clean)
    clean=true
    shift # past argument
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
echo -e "${ep}      install_pack  = ${install_pack}"
echo -e "${ep}      clean         = ${clean}"
echo -e "${ep}"
### END parse arguments ###

# If verbose redirect to stdout, else /dev/null
if [[ "$verbose" = true ]]; then
    out=/dev/stdout
fi

# Check if env var is set and weka.jar could be found
if [[ -z "$WEKA_HOME" ]]; then
    echo -e "${ep}${red}WEKA_HOME env variable is not set!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
elif [[ ! -e "$WEKA_HOME/weka.jar" ]]; then
    echo -e "${ep}${red}WEKA_HOME=${WEKA_HOME} does not contain weka.jar!" > /dev/stderr
    echo -e "${ep}Exiting now...${nc}" > /dev/stderr
    exit 1
fi

export CLASSPATH=${WEKA_HOME}/weka.jar
echo -e "${ep}Classpath = " ${CLASSPATH}


base="./"

pack_name="rnn-forecaster"
zip_name=${pack_name}-${version}.zip

# Clean up lib folders and classes
if [[ "$clean" = true ]]; then
    [[ -d lib ]] && rm lib/* &> ${out}
    mvn -q clean > /dev/null # don't clutter with mvn clean output
fi

# Compile source code with maven
echo -e "${ep}Pulling dependencies via maven..."
mvn -q -DskipTests=true install >  ${out}

echo -e "${ep}Starting ant build for ${bold}"${base}${nc}

# Clean-up
ant -f build_package.xml clean > /dev/null # don't clutter with ant clean output

# Build the package
ant -f build_package.xml make_package > ${out}

# Install package from dist dir
if [[ "$install_pack" = true ]]; then
    # Remove up old packages
    if [[ "$clean" = true ]]; then
        [[ -d ${WEKA_HOME}/packages/${pack_name} ]] && rm -r ${WEKA_HOME}/packages/${pack_name} &> ${out}
    fi
    echo -e "${ep}Installing ${pack_name} package..."
    java -cp ${CLASSPATH} weka.core.WekaPackageManager -install-package dist/${zip_name} > ${out}
    if [ $? -eq 0 ]; then
        echo -e "${ep}Installation successful"
    else
        echo -e "${ep}Installation failed"
    fi
fi
