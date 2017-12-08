all :
	pdflatex -output-directory=out project.tex && bibtex out/project.aux && pdflatex -output-directory=out project.tex && pdflatex -output-directory=out project.tex && open out/project.pdf

clean:
	rm out/*
