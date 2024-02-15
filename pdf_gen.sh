#!/bin/bash

# Generate the PDF
cd .. && jb build math-5650-6650 --builder latex && cd math-5650-6650

# Define the file path
FILEPATH="./_build/latex/book.tex"

# Define the search string
SEARCH_STRING="Bjarne"

# Define the string to insert
INSERT_STRING="\\\\\setmainfont[ Path = ../../font/, UprightFont = *-Regular,BoldFont = *-Bold,ItalicFont = *-Italic]{GentiumBookPlus}\\\\setsansfont[ Path = ../../font/, UprightFont = *-Regular,BoldFont = *-Bold,ItalicFont = *-Italic]{GentiumBookPlus}"

# Search for the string in the file
if grep -q "$SEARCH_STRING" "$FILEPATH"; then
    # Insert the string before the matched line
    sed -i "/$SEARCH_STRING/i $INSERT_STRING" "$FILEPATH"
    echo "String inserted successfully."
else
    echo "String not found in the file."
fi

cd _build/latex && make && cd ..