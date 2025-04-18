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
