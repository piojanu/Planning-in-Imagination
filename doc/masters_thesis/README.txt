
 ------ Struktura plik�w
Rozdzia�y wrzucane powinny by� do katalogu 'rozdzialy' i dodawane z poziomu g��wnego dokumentu poprzez linijk�:
\input{rozdzialy/plik.tex} \newpage

oczywi�cie jest to tylko umowna konwencja i mo�na to zmieni�, ale ten szablon ju� tak� przyj��. Nazwa pliku te� nie musi mie� nic wsp�lnego z nazw� rozdzia�u.
Szablon zak�ada te� istnienie r�nych plik�w dodatkowych (g��wnie opisuj�cuch strony, kt�re s� wymagane w my�l wymaga� edytorskich) w podkatalogu meta. S� to odpowiednio:
stresznie.tex
abstract.tex
spis_tresci.tex
wykaz_wazniejszych_oznaczen_i_skrotow.tex

dodatkowo w katalogu meta powinna si� znajdowa� bibliografia w pliku
bibliografia.bib

 ------ R�ne operacja
Ka�dy rozdzia� rozpoczynamy od \section, podrozdzia� od \subsection i podpodrozdzia� od \subsubsection - inne elementy struktury dokumentu (np. \chap) nie s� obs�ugiwane

Nowy paragraf poprzez podw�jny enter (nie nale�y u�ywa� \paragraph{} itd)

wstawianie obrazka:

\begin{figure}[H]
\includegraphics[width,scale,itd.]{plik.png}
\caption{Podpis obrazka}
\label{obrazek_1}
\end{figure}

Przywo�anie w tekscie:

Blabla przedstawia \ref{obrazek_1}

Wstawienie tabeli:

\begin{table}[H]
\caption{Podpis tabeli}
\label{tabela_1}
\begin{tabu} to \textwidth {| p | p | ... |}
\hline
\rowcolor{lightgray} col1_header & col2_header & ... \\ \hline
col1_row1 & col2_row1 & ... \\ \hline
col1_row2 & col2_row2 & ... \\ \hline
\end{tabu}

Przywo�anie jak dla obrazka, tylko wpisujemy w \ref to samo co mamy w \label

wstawianie tabeli na wi�cej ni� jedn� stron�:

\begin{longtabu} to \textwidth {| p | p | ... |}
\caption{Podpis tabeli} \label{tabela_2} \\[-2mm] %-2mm jest wymagane, poniewaz caption w longtabu jest uznawany jako wiersz i dzia�a na niego ustawione ograniczenie min. 2mm odst�pu przed nast�pnym wierszem
\hline
\rowcolor{lightgray} col1_header & col2_header & ... \\ \hline
\endfirsthead

\rowcolor{lightgray} col2_header & col2_header & ... \\ \hline
\endhead

col1_row1 & col2_row1 & ... \\ \hline
col1_row2 & col2_row2 & ... \\ \hline
\end{longtabu}
