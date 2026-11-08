LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIB   = bibtex

all: main.pdf

main.pdf: main.tex $(wildcard sections/*.tex) refs.bib $(wildcard figs/*.png) $(wildcard figs/*.jpg)
	$(LATEX) main.tex
	$(BIB) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf

.PHONY: all clean
