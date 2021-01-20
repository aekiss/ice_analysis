#!/usr/bin/env sh

# create a local copy of figures

read -p "NCI username: " uname

rsync -aSH --progress --partial-dir=.rsync-partial --no-perms --no-owner --no-group --update --exclude '_*' --exclude '.*' --exclude '.*/'  $uname@gadi-dm.nci.org.au:/g/data/v45/aek156/notebooks/github/aekiss/ice_analysis/figs . || exit 1

echo 'done'
exit 0
