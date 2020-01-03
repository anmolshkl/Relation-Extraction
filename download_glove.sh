#!/bin/bash

set -e
set -x

GLOVE_URL=http://nlp.stanford.edu/data/glove.6B.zip

cd data/
wget $GLOVE_URL
unzip $(basename $GLOVE_URL)
rm $(basename $GLOVE_URL)
