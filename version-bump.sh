#!/usr/bin/env bash

old=$1
new=$2

sed -i.bak -e "0,/${old}/ s/${old}/${new}/" pom.xml
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" build_package.xml

git add pom.xml -v
git add build_package.xml -v
git commit -m "Version bump v${old} to v${new}"
