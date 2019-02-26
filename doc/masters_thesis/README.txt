
 ------ Struktura plików
Rozdzia³y wrzucane powinny byæ do katalogu 'rozdzialy' i dodawane z poziomu g³ównego dokumentu poprzez linijkê:
\input{rozdzialy/plik.tex} \newpage

oczywiœcie jest to tylko umowna konwencja i mo¿na to zmieniæ, ale ten szablon ju¿ tak¹ przyj¹³. Nazwa pliku te¿ nie musi mieæ nic wspólnego z nazw¹ rozdzia³u.
Szablon zak³ada te¿ istnienie ró¿nych plików dodatkowych (g³ównie opisuj¹cuch strony, które s¹ wymagane w myœl wymagañ edytorskich) w podkatalogu meta. S¹ to odpowiednio:
stresznie.tex
abstract.tex
spis_tresci.tex
wykaz_wazniejszych_oznaczen_i_skrotow.tex

dodatkowo w katalogu meta powinna siê znajdowaæ bibliografia w pliku
bibliografia.bib

 ------ Ró¿ne operacja
Ka¿dy rozdzia³ rozpoczynamy od \section, podrozdzia³ od \subsection i podpodrozdzia³ od \subsubsection - inne elementy struktury dokumentu (np. \chap) nie s¹ obs³ugiwane

Nowy paragraf poprzez podwójny enter (nie nale¿y u¿ywaæ \paragraph{} itd)

wstawianie obrazka:

\begin{figure}[H]
\includegraphics[width,scale,itd.]{plik.png}
\caption{Podpis obrazka}
\label{obrazek_1}
\end{figure}

Przywo³anie w tekscie:

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

Przywo³anie jak dla obrazka, tylko wpisujemy w \ref to samo co mamy w \label

wstawianie tabeli na wiêcej ni¿ jedn¹ stronê:

\begin{longtabu} to \textwidth {| p | p | ... |}
\caption{Podpis tabeli} \label{tabela_2} \\[-2mm] %-2mm jest wymagane, poniewaz caption w longtabu jest uznawany jako wiersz i dzia³a na niego ustawione ograniczenie min. 2mm odstêpu przed nastêpnym wierszem
\hline
\rowcolor{lightgray} col1_header & col2_header & ... \\ \hline
\endfirsthead

\rowcolor{lightgray} col2_header & col2_header & ... \\ \hline
\endhead

col1_row1 & col2_row1 & ... \\ \hline
col1_row2 & col2_row2 & ... \\ \hline
\end{longtabu}
