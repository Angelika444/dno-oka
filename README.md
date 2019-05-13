# Wykrywanie-naczyn-dna-oka
Koncepcja algorytmu w podstawowej wersji:
1) obraz wejściowy jest rozjaśniany, nasycane są barwy
2) wykrywane jest koło na zdjęciu, czyli zarys oczodołu
3) proporcjonalnie zwiększane są wartości koloru zielonego dla wszystkich pikseli-daje większy kontrast między czerwonymi i pomarańczowymi pikselami
4) wyostrzane są krawędzie na obrazie
5) stosowana jest maska przepuszaczająca tylko barwę czerwoną
6) uśrednienie wartości pikseli i nadanie tej barwy tłu-żeby nie było kontrastu
7) usunięcie poziomych linii krawędzi oczodołu, ponieważ oczodół to "ścięte koło"
8) zastosowanie filtru Laplace'a
9) wykrywanie krawędzi-Canny
10) zamkniecie krawędzi + erozja

Zdjęcie końcowe:
- kolor niebieski-naczynka prawidłowo wykryte
- kolor zielony-błędnie wykryte naczynka w miejscu, gdzie ich oryginalnie nie ma
- kolor czerwony-niewykryte naczynka
