# Dokumentacja Danych Eksperymentu

## Przegląd
Ten dokument opisuje strukturę danych CSV używaną do przechowywania wyników eksperymentu MIT/MOT. 

## Struktura Danych
Każdy wpis w zestawie danych zawiera następujące pola:

### Informacje o Uczestniku
- **ID**: Identyfikator uczestnika, wprowadzony ręcznie przez eksperymentatora
- **Sex**: Płeć osoby badanej ["Nie chcę podawać", "Kobieta", "Mężczyzna"]
- **Age**: Wiek badanego

### Struktura Eksperymentu
- **Block**: Blok z którego pochodzi próba, numerowane od 1, bloki treningowe również są liczone
- **Trial**: Numer próby w danym bloku, numerowane od 1
- **Type**: Typ próby ["MIT", "MOT"]
- **Is_training**: Wskaźnik binarny [1,0], gdzie 1 = próba treningowa, 0 = próba z badania
- **N_targets**: Liczba targetów w próbie
- **N_distractors**: Liczba dystraktorów w próbie
- **N_circles**: Liczba orbit w próbie (N_circles*2 = N_targets + N_distractors)
- **Timestamp**: Czas wyświetlenia instrukcji "wciśnij przycisk aby rozpocząć następną próbę..." przed próbą z danego wyniku

### Dane związane z wyborem target/dystraktor

- **Ground_truth_guess**: ["target", "distractor"] - Informacja o tym jakiego typu obiekt jest oznaczony do zgadnięcia
- **Guess**: ["target", "distractor", -1] - Informacja o tym jakiego typu obiekt został wybrany przez badanego; -1 oznacza że został przekroczony czas przeznaczony na zadanie
- **Guess_success**: [1, 0, -1] - Wskaźnik sukcesu:
  - 1: jeśli 'Guess' == 'Ground_truth_guess'
  - 0: jeśli 'Guess' != 'Ground_truth_guess'
  - -1: jeśli 'Guess' = -1 (próba porzucona)
- **Task_time_guess**: Czas wykonywania zadania wyboru między targetem/dystraktorem w sekundach

### Identyfikcja kształtu w zadaniu MIT

- **MIT_obj_identified**: [0, 1, 2, 3, -1, "MOT"] - Wynik identyfikacji obiektu:
  - 0: nie wskazano "dystraktora" oraz jeśli kształt targetu jest różny od wskazanego kształtu (przy czym oznaczony obiekt w trialu jest targetem) - odpowiedz zła
  - 1: nie wskazano "dystraktora" oraz jeśli kształt targetu == kształt wskazany (przy czym oznaczony obiekt w trialu jest targetem) - odpowiedz prawidłowa
  - 2: jeśli wskazano "dystraktor" i oznaczony obiekt jest dystraktorem - odpowiedz prawidłowa
  - 3: jeśli wskazano "dystraktor", a oznaczony obiekt nie jest dystraktorem - odpowiedz zła
  - 4: nie wskazano "dystraktora" oraz jeśli kształt oznaczonego obiektu == kształt wskazany (przy czym oznaczony obiekt w trialu jest dystraktorem) - odpowiedz zła (chociaż wskazano dobry kształt)
  - 5: nie wskazano "dystraktora" oraz jeśli kształt oznaczonego obiektu jest różny od wskazanego kształtu (przy czym oznaczony obiekt w trialu jest dystraktorem) - odpowiedz zła
  - -1: przekroczono czas przeznaczony na zadanie bądź próba została wyrzucona we wcześniejszym zadaniu (Guess_success == -1)
  - "MOT": nie dotyczy
- **Task_time_identification**: [czas w sekundach, "MOT", -1] - Czas wykonania zadania wskazywania kształtu, "MOT" jeśli nie dotyczy bo zły typ zadania, -1 jeśli próba została wcześniej wywalona (Guess_success == -1)

### Miary Odpowiedzi Motorycznej
- **TargetX**: Koordynata X celu, -1 jeśli zadanie zostało pominięte (brak wciśniętego klawisza)
- **TargetY**: Koordynata Y celu, -1 jeśli zadanie zostało pominięte (brak wciśniętego klawisza)
- **ClickX**: Koordynata X kliknięcia, -1 jeśli zadanie zostało pominięte lub przekroczony został czas na zadanie
- **ClickY**: Koordynata Y kliknięcia, -1 jeśli zadanie zostało pominięte lub przekroczony został czas na zadanie
- **Norm_Euc_Dist**: Norma euklidesowska, znormalizowana do wielkości ekranu, dana wzorem:
  ```
  math.sqrt(pow((objective.pos[0] - click_pos[0])/self.win.size[0], 2) + pow((objective.pos[1] - click_pos[1])/self.win.size[1], 2))
  ```
  Wartość wynosi -1 jeśli zadanie zostało pominięte lub przekroczony został czas na zadanie

- **Task_time_motoric**: Czas (w sekundach) zadania od momentu kiedy kółko zaczęło się kręcić do końca, -1 jeśli zadanie zostało pominięte
- **Movement_start**: Czas, liczony od początku zadania, kiedy osoba badana puściła przycisk i zaczęła ruch, -1 jeśli zadanie zostało pominięte lub przekroczony został czas na zadanie
- **Movement_duration**: Czas w sekundach trwania ruchu (od puszczenia klawisza do naciśnięcia ekranu), gdzie Task_time_motoric = Movement_start + Movement_duration, -1 jeśli zadanie zostało pominięte lub przekroczony został czas na zadanie

### Klikniecie przed/za celem:
- **Motoric_obj_Vx** Koordynata X wektoru opisującego ruch obieku
- **Motoric_obj_Vy** Koordynata Y wektoru opisującego ruch obieku
- **Motoric_obj_V1_magnitude** Wartość wektoru prędkości obiektu
- **Motoric_click_V2_magnitude** Wartość wektoru między obiektem, a klikniętym punktem (w praktyce odległość kliknięcia od obiektu z kierunkiem)
- **Angle_objV_click** Kąt między wktorami V1 i V2, w praktyce wartości [0,90] oznaczją kliknięcie 'PRZED' obiektem względem prędkości, zaś [90,180] 'ZA'

### Informacje o Bodźcach Wizualnych
- **Indicated_img**: 
  - "MOT" (dotyczy tylko MIT)
  - -1 (jeśli 'Guess_success' == -1)
  - 0 (jeśli czas na zadanie identyfikcji kształtu przekroczony)
  - "distractor" (jeśli wskazany dystraktor)
  - "xxx.png" (jeśli wskazany konkretny kształt)
- **Img_to_guess**: 
  - "MOT" (dotyczy tylko MIT)
  - "xxx.png" (kształt który był oznaczony w zadaniu)

### Informacje które .png były targetem/distractorem/niewykorzystane
- Każde zdjęcie z eksperymentu ma swoją kolumnę i w niej wartość ("MOT",-1,1,2):
    - "MOT" (dotyczy tylko MIT)
    - 1 (nie wykorzystane w próbie MIT)
    - 1 (target)
    - 2 (distractor)

## Uwagi
- Wartość -1 zazwyczaj oznacza, że zadanie zostało pominięte, porzucone lub przekroczony został limit czasu
- Zadania MIT i MOT mają niektóre pola specyficzne dla każdego typu zadania

## Miary Kinematyczne (z *performance_kinematics_measures*)

### Miary końcówki palca (zadanie dotykowe)
- **MT** *(Movement Time; s)* — czas od puszczenia klawisza do dotknięcia ekranu (dokładność do tysięcznych sekundy).
- **TE** *(Touching Error; px)* — odległość od środka ruchomego kółka (z zadania motorycznego) do punktu dotknięcia ekranu.
- **PoL** *(Predicting or Lagging; {-1, 1})* — wskaźnik wyprzedzania lub pozostawania w tyle względem kierunku ruchu kółka:
  - **1** — dotknięcie „przed” obiektem (w przedniej połowie względem wektora ruchu),
  - **-1** — dotknięcie „za” obiektem (w tylnej połowie względem wektora ruchu).
- **PV** *(Peak Velocity; cm/s)* — maksymalna prędkość liniowa końcówki palca.
- **P2PV** *(% czasu)* — czas do osiągnięcia szczytowej prędkości jako **procent** całkowitego czasu ruchu (od puszczenia klawisza do dotknięcia ekranu).
- **AV** *(Average Velocity; cm/s)* — średnia prędkość liniowa.
- **D2TPV** *(Distance to Target at Peak Velocity; %)* — dystans do celu w momencie osiągnięcia PV jako **procent** całkowitej długości ruchu.
- **D2TEM** *(Distance to Target near End Movement; %)* — jak wyżej, ale mierzony w momencie, gdy prędkość spada poniżej **10%** wartości PV (po osiągnięciu PV).
- **XYPV** *(cm, cm)* — współrzędne **X, Y** w momencie PV.
- **XYEM** *(cm, cm)* — współrzędne **X, Y** w momencie spadku prędkości poniżej **10%** PV.

### Kodowanie sukcesu (dokładność)
- **MOT** — 1 jeśli poprawnie wskazano, czy po okresie śledzenia wskazany był target czy dystraktor; w przeciwnym razie 0.
- **MIT** — 1 tylko wtedy, gdy:
  - kliknięto **target** (zielone kółko) **i** poprawny **kształt** na wianku, *albo*
  - kliknięto **dystraktor** (czerwone kółko) **i** odpowiadający mu **symbol** na wianku.
  W pozostałych przypadkach 0.

> **Raportowanie:** Poziom wykonania liczymy jako **% poprawnych odpowiedzi** w danym warunku (MOT: 3 poziomy trudności; MIT: 3 poziomy trudności) — osobno dla każdego badanego. Dodatkowo dla każdej z miar powyżej raportujemy **odchylenie standardowe** w obrębie warunku jako wskaźnik zmienności.

### Elipsa ufności dla punktów dotknięcia
Dla każdego warunku wyznaczamy elipsę obejmującą końcowe punkty dotknięcia ekranu (przyjmując, że centrum ruchomego kółka to *(0, 0)*):
- **Centrum** — średnia z końcowych współrzędnych dotknięć.
- **Półosie** — pierwiastki kwadratowe z **wartości własnych** macierzy kowariancji (obliczonej dla odchyleń końcowych pozycji względem średniej).
- **Orientacja** — wektory własne macierzy kowariancji.
- **Miara** — **pole elipsy** (jako globalny wskaźnik precyzji/rozrzutu).

---

## Miary kątowe ramienia i łokcia

**Definicje kątów:**
- **Shoulder elevation** — kąt (°) między tułowiem a ramieniem; gdy ramię wzdłuż tułowia, kąt ≈ 0°.
- **Elbow flexion** — kąt (°) po wewnętrznej stronie łokcia między ramieniem a przedramieniem.

**W każdej próbie** obliczamy (dla obu kątów osobno):
- **PV** *(°/s)* — szczytowa prędkość kątowa.
- **P2PV** *(% czasu)* — czas do PV w % całkowitego czasu ruchu.
- **AV** *(°/s)* — średnia prędkość kątowa.
- **ROM** *(°)* — zakres ruchu (max – min).
- **AA** *(°)* — średni kąt.
- **SA** *(°)* — kąt początkowy.
- **AUMC** *(obszar pod krzywą ruchu; s·°)* — całka z przebiegu kąta po czasie (pole pod krzywą kąt–czas).
- **AvgPh** *(—)* — średnia faza z całej próby, gdzie faza w każdym punkcie czasu wyznaczana jest z unormowanych do maksimum: kąta i prędkości kątowej.

> **Raportowanie zmienności:** Dla PV, P2PV, AV, ROM, AA i AUMC — w każdym warunku wyliczamy **odchylenie standardowe** (wariabilność), osobno dla każdej osoby.

### Time-normalized miary
- **TNA** *(Time Normalized Angle)* — normalizujemy czas próby do 100% i dzielimy na koszyki (np. co **20%** punktów czasu). W każdym koszyku liczymy średni **kąt** z prób danego warunku, a następnie **SD** tych średnich po koszykach.
- **TNP** *(Time Normalized Phase)* — analogicznie do TNA, ale dla **fazy**.

### Koordynacja ramię–łokieć
- **TLPV** *(Time Lag to Peak Velocity; %)* — różnica (w % całkowitego czasu ruchu) między chwilą osiągnięcia PV przez ramię a łokieć.
- **CCJA** *(Cross-Correlation of Joints’ Angles)* — współczynnik korelacji Pearsona między przebiegami kątów ramienia i łokcia przy zerowym przesunięciu.
- **ACRP** *(Average Continuous Relative Phase; —)* — średnia z **CRP(t) = phase(shoulder) − phase(elbow)**; opcjonalnie również w wariancie **time-normalized** jak wyżej.

### Struktura zmienności kinematycznej
- **SaEn** *(Sample Entropy)* — miara złożoności szeregu czasowego; liczona na przebiegach kątów lub fazy.
- **RQA** *(Recurrence Quantification Analysis)* — miary oparte na rekureencji do oceny regularności i innych własności sygnału (do doprecyzowania parametrów obliczeń przed implementacją).

---

## Agregacja i raportowanie
- Wszystkie miary raportujemy **osobno dla każdego badanego** i **dla każdego warunku** (6 warunków: 3 × MOT, 3 × MIT).
- Dla miar binarnych (sukces MOT/MIT) podajemy **% poprawnych odpowiedzi**.
- Dla miar ciągłych raportujemy **średnią** i **odchylenie standardowe** (wariabilność) w obrębie warunku.
