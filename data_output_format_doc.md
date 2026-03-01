# Dokumentacja Eksperymentu MIT/MOT

## Opis Badania

### 1. Cel badania

Badanie ma na celu pomiar zdolności uwagowego śledzenia wielu obiektów (**MOT** — Multiple Object Tracking) oraz śledzenia wielu tożsamości (**MIT** — Multiple Identity Tracking) w powiązaniu z koordynacją wzrokowo-motoryczną. Każda próba łączy zadanie percepcyjne z zadaniem motorycznym, co pozwala jednocześnie ocenić zarówno sprawność uwagi wzrokowej, jak i precyzję oraz dynamikę ruchu kończyny górnej.

### 2. Uczestnicy

Przed rozpoczęciem eksperymentu każdy uczestnik wypełnia formularz demograficzny (PsychoPy GUI), podając:
- **ID** uczestnika
- **Płeć** (Kobieta / Mężczyzna / Nie chcę podawać)
- **Wiek**

### 3. Aparatura

- **Oprogramowanie:** PsychoPy (Python)
- **Ekran:** monitor dotykowy, tryb pełnoekranowy
- **Kamera internetowa:** 1280×720 px, 60 fps (MJPEG), uruchamiana automatycznie w trakcie zadania motorycznego — rejestruje obraz uczestnika w celu późniejszej estymacji pozy ciała (3D pose estimation, model RTMPose3D / RTMDet)
- **Urządzenie wejściowe:** ekran dotykowy + dedykowany klawisz, na którym uczestnik trzyma palec środkowy między zadaniami

### 4. Warunki eksperymentalne

Eksperyment składa się z dwóch typów zadań realizowanych w ramach oddzielnych bloków:

| Parametr | MOT | MIT |
|---|---|---|
| **Obiekty** | Identyczne czarne koła | Unikalne kształty (obrazki PNG — pary a/b, 11 zestawów) |
| **Liczba celów (N_targets)** | 2, 3, 4 | 1, 2, 3 |
| **Liczba kół satelitarnych** | 5 | 5 |
| **Liczba dystraktorów** | 2×5 − N_targets | 2×5 − N_targets |
| **Identyfikacja tożsamości** | Nie | Tak (dodatkowe zadanie) |

Obciążenie poznawcze (Load) definiowane jest jako:
- **LOW:** MIT_1 / MOT_2
- **MID:** MIT_2 / MOT_3
- **HIGH:** MIT_3 / MOT_4

### 5. Struktura sesji eksperymentalnej

1. **Blok treningowy:** 12 prób MIT + 12 prób MOT (kolejność losowa)
2. **Bloki eksperymentalne:** 2 pary bloków × 60 prób = **240 prób** (120 MIT + 120 MOT)
   - Kolejność bloków (MIT-first vs MOT-first) jest losowana per uczestnik
   - Wewnątrz bloku rozkład wariantów trudności (N_targets) jest zrównoważony i randomizowany
3. **Przerwy:** Między blokami uczestnik może odpocząć

### 6. Przebieg pojedynczej próby

Każda próba rozpoczyna się od wciśnięcia przez uczestnika klawisza „X” na dedykowanej klawiaturze (przy użyciu palca środkowego prawej dłoni) i składa się z 8 następujących po sobie faz:

###### Faza 1 — Obserwacja (1.5 s)
Obiekty (cele oznaczone zielonymi obwódkami oraz dystraktory) są widoczne na ekranie i nieruchome. Zadaniem osoby badanej jest zapamiętanie, które obiekty są celami.
*   *W warunku MIT:* obiekty posiadają unikalne kształty widoczne w tej fazie.
*   *W warunku MOT:* wszystkie obiekty mają identyczny wygląd (czarne koła).

###### Faza 2 — Śledzenie (6 s)
Oznaczenia celów (zielone obwódki) znikają. W warunku MIT unikalne kształty **pozostają widoczne** — obiekty poruszają się w swojej pełnej formie. W warunku MOT wszystkie obiekty wyglądają identycznie (czarne koła). Zadaniem uczestnika jest ciągłe śledzenie wzrokiem przemieszczających się celów. Wszystkie obiekty poruszają się po złożonych trajektoriach:
*   Obiekty krążą parami wokół niewidocznych kół satelitarnych (częstotliwość: **0.45 Hz**).
*   Koła satelitarne krążą wokół centralnego punktu (częstotliwość: **0.1 Hz**).
*   W losowym momencie (z przedziału 1 s – 6 s) każde koło satelitarne może jednorazowo odwrócić kierunek obrotu.
*   Kierunki obrotu, offsety kątowe i orientacje par obiektów są randomizowane.

###### Faza 3 — Zakrycie i oznaczenie (1.5 s)
Obiekty zatrzymują się. W warunku MIT następuje **zakrycie wszystkich obiektów identycznymi czarnymi kołami** (ukrycie tożsamości). Następnie jeden losowo wybrany element zostaje podświetlony niebieską obwódką. Zadaniem osoby badanej jest przypomnienie sobie — na podstawie wcześniejszego śledzenia — czy podświetlony obiekt był celem, czy dystraktorem.

###### Faza 4 — Oczekiwanie (2.3 s)
Wszystkie obiekty znikają, a na ekranie widoczne jest jedynie szare tło (w parametrach zdefiniowane jako opóźnienie `camera_pause`). Klawisz „X” musi pozostać wciśnięty – zbyt wczesne puszczenie przycisku w tej fazie powoduje, że zadanie motoryczne nie pojawia się na ekranie.

###### Faza 5 — Zadanie motoryczne (Limit czasu: 10 s)
Na ekranie pojawia się docelowy obiekt motoryczny – osobne czarne koło z białym krzyżykiem (+), poruszające się po okręgu (częstotliwość: **0.35 Hz**, promień: ~2× promień kół satelitarnych). Kierunek ruchu obiektu zmienia się losowo raz w trakcie próby.
Zadaniem uczestnika jest:
1.  Zwolnić trzymany klawisz „X” palcem środkowym (rejestracja zmiennej **Movement_start**).
2.  Wyciągnąć ramię i płynnym ruchem dotknąć ekranu w miejscu poruszającego się obiektu (rejestracja współrzędnych **ClickX, ClickY**).
W momencie puszczenia klawisza, kamera automatycznie rozpoczyna nagrywanie ruchu na potrzeby estymacji pozy 3D.
Mierzone są m.in. następujące zmienne:
*   Czas reakcji przedmotorycznej (`Movement_start`)
*   Czas trwania ruchu dłoni (`Movement_duration`)
*   Znormalizowany dystans euklidesowy / błąd celowania (`Norm_Euc_Dist` / `TE`)
*   Wektory i kąty uderzenia względem ruchu obiektu (`Motoric_obj_Vx`, `Motoric_obj_Vy`, `Angle_objV_click`)

###### Faza 6 — Kategoryzacja: Cel czy Dystraktor (Limit czasu: 4 s)
Na ekranie pojawiają się dwa koła: zielone (reprezentujące cel) i czerwone (reprezentujące dystraktor). Ich pozycja (lewo/prawo) jest losowana. Uczestnik, poprzez dotknięcie odpowiedniego symbolu, kategoryzuje obiekt, który był oznaczony niebieską obwódką w Fazie 3.
*   Rejestrowane zmienne: `Ground_truth_guess`, `Guess`, `Guess_success`, `Task_time_guess`.

###### Faza 7 — Identyfikacja tożsamości (tylko MIT, Limit czasu: 8 s)
Wszystkie unikalne kształty użyte w danej próbie zostają odkryte i rozmieszczone losowo na okręgu. U góry ekranu wyświetla się dodatkowo czarne koło reprezentujące „dystraktor”. Uczestnik ma za zadanie wskazać dokładną tożsamość ukrytą pod obiektem z Fazy 3:
*   Poprawne wskazanie konkretnego kształtu (jeśli oznaczono cel) = sukces (`MIT_obj_identified = 1`).
*   Poprawne wskazanie czarnego koła (jeśli oznaczono dystraktor) = sukces (`MIT_obj_identified = 2`).
*   Wskazania błędne (np. zły kształt lub pomylenie celu z dystraktorem) są rejestrowane jako błędy (`MIT_obj_identified = 0, 3, 4, 5`).

###### Faza 8 — Informacja zwrotna
Na ekranie wyświetlana jest informacja o poprawności wykonania zadań z faz kategoryzacji oraz identyfikacji. Naciśnięcie klawisza „X” pozwala pominąć ten komunikat, co płynnie inicjuje rozpoczęcie kolejnej próby eksperymentalnej.

### 7. Parametry ruchu obiektów

| Parametr | Wartość |
|---|---|
| FPS | 60 |
| Promień głównego koła (`circle_radius`) | 286.72 px |
| Promień kół satelitarnych (`small_circle_radius`) | 114.69 px |
| Promień obiektów (`obj_radius`) | 32.77 px |
| Częstotliwość obrotu obiektów (`hz_target`) | 0.45 Hz |
| Częstotliwość obrotu kół satelitarnych (`hz_circle`) | 0.1 Hz |
| Częstotliwość obrotu obiektu motorycznego (`hz_motoric`) | 0.35 Hz |
| Promień obiektu motorycznego (`motoric_radius`) | 98.30 px |
| Promień orbity motorycznej (`motoric_circle_radius`) | 229.38 px |
| Czas obserwacji (`observation_time`) | 1.5 s |
| Czas śledzenia (`tracking_time`) | 6 s |
| Czas oznaczenia obiektu (`guessing_time`) | 1.5 s |
| Limit czasu — zadanie motoryczne (`motor_task_time_limit`) | 10 s |
| Limit czasu — kategoryzacja cel/dystraktor (`answer_1_time_limit`) | 4 s |
| Limit czasu — identyfikacja kształtu MIT (`answer_MIT_time_limit`) | 8 s |
| Opóźnienie startu kamery (`camera_pause`) | 2.3 s |
| Zmiana kierunku kół satelitarnych (`direction_changes`) | 1× losowo w przedziale [1 s, 6 s] |
| Zmiana kierunku obiektu motorycznego (`direction_changes_motoric`) | 1× losowo w przedziale [1 s, 10 s] |
| Zmiana kierunku głównego koła (`change_big_direction`) | Nie |
| Losowe kierunki kół satelitarnych (`random_direction_small_circles`) | Tak |
| Losowy kierunek głównego koła (`random_direction_big_circle`) | Tak |
| Losowy offset target/dystraktor (`random_offset_target_distractor`) | Tak |
| Losowy offset kół (`random_offset_circles`) | Tak |
| Losowa orientacja par (`random_distractor_target_orientation`) | Tak |
| Tryb obrazków MIT (`img_mode`) | Tak (pary a/b, 11 zestawów kształtów) |
| Wyświetlanie kół satelitarnych (`show_circles`) | Nie |

### 8. Post-processing

Po zebraniu danych przeprowadzana jest automatyczna analiza obejmująca:
1. **Estymacja pozy ciała 3D** (RTMPose3D + RTMDet) z nagrań kamer — ekstrakcja kątów łokciowych, ramiennych, metryk kinematycznych (PV, AV, MT, SaEn, PathLen itd.)
2. **Filtracja sygnału kinematycznego:** Surowe trajektorie pozycji palca (X, Y) oraz kąty stawowe (shoulder elevation, elbow flexion) są wygładzane dolnoprzepustowym filtrem **Butterwortha 2. rzędu z zerowym przesunięciem fazowym** (`scipy.signal.filtfilt`, efektywny rząd = 4, częstotliwość odcięcia = **6 Hz**) przed jakimkolwiek różniczkowaniem numerycznym. Zastosowanie filtracji zero-fazowej eliminuje szum pomiarowy bez przesuwania fizjologicznych punktów zwrotnych w dziedzinie czasu, co jest szczególnie istotne dla rzetelnej estymacji prędkości szczytowej (PV) i czasu jej osiągnięcia (P2PV) (Crenna, Rossi i Berardengo, 2021).
3. **Detekcja anomalii kinematycznych:** zamrożone klatki (>5% bez ruchu), teleportacje (prędkość > median + 5×IQR, min. 50 cm/s), niestabilność długości kości (CV > 0.30)
4. **Wykluczenie prób behawioralnych:** Z analiz wykluczane są:
   - Próby treningowe (`Is_training == 1`)
   - Próby z anomaliami kinematycznymi (`is_anomaly == 1`)
   - Próby z przekroczeniem limitu czasu w zadaniu kategoryzacji (`Guess_success == -1`)
   - Próby z przekroczeniem limitu czasu lub brakiem ruchu w zadaniu motorycznym (`Movement_duration == -1`)
   - Próby z przekroczeniem limitu czasu w zadaniu identyfikacji kształtu MIT (`MIT_obj_identified == -1`)
5. **Filtrowanie outlierów kliknięć:** dystans kliknięcia > 2 × promień obiektu motorycznego (197 px). Zastosowano wyłącznie próg oparty na stałej fizycznej (2 × `motoric_radius`), rezygnując z filtracji statystycznej (3×SD), aby uniknąć systematycznego obcinania triali z warunków o wyższym obciążeniu poznawczym, w których większy rozrzut kliknięć może odzwierciedlać rzeczywisty efekt eksperymentalny.
6. **Analizy statystyczne:** ANOVA (repeated measures, Type × Load), t-testy, korelacje Spearmana, regresja logistyczna, korekta FDR (Benjamini-Hochberg), elipsy ufności dyspersji kliknięć

---

## Dokumentacja Danych CSV

### Przegląd
Ten dokument opisuje strukturę danych CSV używaną do przechowywania wyników eksperymentu MIT/MOT.
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

