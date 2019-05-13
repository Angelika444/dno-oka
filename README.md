# Wykrywanie-naczyn-dna-oka
Koncepcja algorytmu w podstawowej wersji:
1) obraz wejściowy jest rozjaśniany, nasycane są barwy
2) wykrywane jest koło na zdjęciu, czyli zarys oczodołu
3) proporcjonalnie zwiększane są wartości koloru zielonego dla wszystkich pikseli-daje większy kontrast między czerwonymi i pomarańczowymi pikselami
4) uśrednienie wartości pikseli wewnątrz oczodołu, i nadanie tej barwy tłu-żeby nie było kontrastu
5) wykrywanie krawędzi-Canny, można przetestować też inne filtry
6) zamkniecie krawędzi, erozja - na razie sprawdzało mi się najlepiej
7) usunięcie poziomych linii krawędzi oczodołu-powinno znacznie poprawić wyniki (tego jeszcze nie ma)

Na razie parametry funkcji mają charakter dosyć losowy.
Nie wiem na ile rozjaśnianie faktycznie pomaga, czy czasami należy przyciemnić?
Do przetestowania jeszcze jak sprawdza się wyostrzenie krawędzi: sharpening(img) i nałożenie maski filtrującej tylko kolor czerwony mask(img)
