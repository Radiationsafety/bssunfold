.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to remove build files"

clean:
	rm -rf build/*

html:
	sphinx-build -b html source build/html
	@echo "Build finished. The HTML pages are in build/html."

dirhtml:
	sphinx-build -b dirhtml source build/dirhtml
	@echo "Build finished. The HTML pages are in build/dirhtml."

singlehtml:
	sphinx-build -b singlehtml source build/singlehtml
	@echo "Build finished. The HTML page is in build/singlehtml."

pickle:
	sphinx-build -b pickle source build/pickle
	@echo "Build finished; now you can process the pickle files."

json:
	sphinx-build -b json source build/json
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	sphinx-build -b htmlhelp source build/htmlhelp
	@echo "Build finished; now you can run HTML Help Workshop with the .hhp file in build/htmlhelp."

qthelp:
	sphinx-build -b qthelp source build/qthelp
	@echo "Build finished; now you can run "qcollectiongenerator" with the .qhcp file in build/qthelp."

devhelp:
	sphinx-build -b devhelp source build/devhelp
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/books"
	@echo "# ln -s $$(pwd)/build/devhelp $$HOME/.local/share/devhelp/books/bssunfold"

epub:
	sphinx-build -b epub source build/epub
	@echo "Build finished. The epub file is in build/epub."

latex:
	sphinx-build -b latex source build/latex
	@echo "Build finished; the LaTeX files are in build/latex."

latexpdf:
	sphinx-build -b latex source build/latex
	@echo "Running LaTeX files through pdflatex..."
	cd build/latex && pdflatex bssunfold.tex
	@echo "pdflatex finished; the PDF file is in build/latex."

text:
	sphinx-build -b text source build/text
	@echo "Build finished. The text files are in build/text."

man:
	sphinx-build -b man source build/man
	@echo "Build finished. The manual pages are in build/man."

changes:
	sphinx-build -b changes source build/changes
	@echo "The overview file is in build/changes."

linkcheck:
	sphinx-build -b linkcheck source build/linkcheck
	@echo "Link check complete; look for any errors in the above output."

doctest:
	sphinx-build -b doctest source build/doctest
	@echo "Testing of doctests in the sources finished, look at the results in build/doctest/output.txt."